import tensorflow as tf
import numpy as np
import dataclasses
import os
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation
from coopland.models.a3c import config_lib


class AgentModel:
    def __init__(self, hparams: config_lib.AgentModelHParams):
        self.hparams = hparams
        self.input_data_size = 4 + 8 + 5  # visibility + corners + exit
        if self.hparams.use_visible_agents:
            # +4 distances to other agents
            self.input_data_size += 4 * (hparams.max_agents - 1)
        self.i_to_direction = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.i_to_direction)}

    def encode_observation(
        self, agent_id, visibility, corners, visible_other_agents, visible_exit
    ):
        full_observation = (
            agent_id,
            visibility,
            corners,
            visible_other_agents,
            visible_exit,
        )
        result = []
        result.extend(visibility)
        result.extend(sum(corners, []))
        exit_vec = [0] * 5
        exit_dir, exit_dist = visible_exit
        if exit_dir is not None:
            exit_vec[self.directions_to_i[exit_dir]] = 1
            exit_vec[-1] = exit_dist
        result.extend(exit_vec)

        if self.hparams.use_visible_agents:
            visible_agents_part = [0, 0, 0, 0] * (self.hparams.max_agents - 1)
            for ag_id, direction, dist in visible_other_agents:
                if ag_id >= self.hparams.max_agents:
                    continue
                offs = 4 * (ag_id if ag_id < agent_id else ag_id - 1)
                if dist == 0:
                    visible_agents_part[offs : offs + 4] = 1, 1, 1, 1
                else:
                    i = offs + self.directions_to_i[direction]
                    visible_agents_part[i] = 1 / dist
            result.extend(visible_agents_part)

        vector = np.array(result)
        assert vector.shape == (self.input_data_size,)
        return (
            np.expand_dims(vector, 0),
            {"input_vector": vector, "full_observation": full_observation},
        )

    def decode_nn_output(self, outputs, metadata, greed_choice_prob=None):
        probs = outputs[0][0, 0]
        value = outputs[1][0, 0]
        input_vector = metadata["input_vector"]
        observation = metadata["full_observation"]
        if greed_choice_prob is not None and np.random.random() < greed_choice_prob:
            direction_i = np.argmax(probs)
        else:
            direction_i = np.random.choice(range(len(probs)), p=probs)
        direction = self.i_to_direction[direction_i]
        return Move(
            direction=direction,
            direction_idx=direction_i,
            input_vector=input_vector,
            probabilities=probs,
            critic_value=value,
            observation=observation,
        )

    def build_layers(self, name=None):
        if name:
            name_prefix = name + "_"
        else:
            name_prefix = ""

        rnn = tf.keras.layers.RNN(
            StackedLSTMCells(
                [tf.keras.layers.LSTMCell(units) for units in self.hparams.rnn_units]
            ),
            return_state=True,
            return_sequences=True,
            name=name_prefix + "RNN",
        )
        actor_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(4), name=name_prefix + "Actor/Head"
        )
        critic_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1), name=name_prefix + "Critic/Head"
        )

        rnn.build((None, None, self.input_data_size))
        actor_head.build((None, None, self.hparams.rnn_units[-1]))
        critic_head.build((None, None, self.hparams.rnn_units[-1]))

        saver = tf.train.Saver(
            [
                *rnn.trainable_variables,
                *actor_head.trainable_variables,
                *critic_head.trainable_variables,
            ]
        )
        return AgentInstance(rnn, actor_head, critic_head, saver)

    def create_agent_fn(
        self, agent_instance: "AgentInstance", session, greed_choice_prob=None
    ):
        states = {}

        def agent_fn(*observation):
            agent_id = observation[0]
            input_data, metadata = self.encode_observation(*observation)

            feed = {input_ph: [input_data]}
            feed.update(zip(prev_states_phs, states.get(agent_id, [])))

            output_data, new_states = session.run(
                ((actor_probabilities_t, critic_t), new_states_t), feed
            )
            states[agent_id] = new_states
            move = self.decode_nn_output(output_data, metadata, greed_choice_prob)
            return move

        def init_before_game():
            states.clear()

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        input_ph = tf.compat.v1.placeholder(tf.float32, [1, None, self.input_data_size])
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_states_t,
            prev_states_phs,
        ] = agent_instance.call(input_ph)
        del actor_logits_t

        return agent_fn


@dataclasses.dataclass
class AgentInstance:
    rnn: "tf.keras.layers.RNN"
    actor_head: "tf.keras.layers.Layer"
    critic_head: "tf.keras.layers.Layer"
    saver: "tf.train.Saver"

    def call(self, input_tensor, sequence_lengths_tensor=None, input_mask=None):
        if input_mask is None:
            if sequence_lengths_tensor is not None:
                input_mask = build_input_mask(sequence_lengths_tensor)

        features, states_after, states_before_phs = _call_stateful_rnn(
            self.rnn, input_tensor, input_mask
        )
        actor_logits = self.actor_head(features)
        critic_value = self.critic_head(features)[:, :, 0]
        actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)

        return (
            actor_logits,
            actor_probabilities,
            critic_value,
            states_after,
            states_before_phs,
        )

    def get_variables(self):
        layers = self.rnn, self.actor_head, self.critic_head
        return [v for layer in layers for v in layer.trainable_variables]

    def save_variables(self, session, directory, step):
        checkpoint_path = os.path.join(directory, "model.ckpt")
        self.saver.save(
            session, checkpoint_path, global_step=step, write_meta_graph=False
        )

    def load_variables(self, session, directory):
        save_path = tf.train.latest_checkpoint(directory)
        self.saver.restore(session, save_path)
        step = int(save_path.rpartition("-")[2])
        return step


@dataclasses.dataclass
class Move:
    direction: Direction
    direction_idx: int
    input_vector: np.ndarray
    probabilities: np.ndarray
    critic_value: np.ndarray
    observation: Observation
    debug_text: str = ""

    def __repr__(self):
        return f"<{self.direction} V={self.critic_value:3f} A={self.probabilities}>"


def _call_stateful_rnn(rnn, input_tensor, input_mask=None):
    if input_tensor.get_shape().as_list()[0] is None:
        batch_size = tf.shape(input_tensor)[0]
        batch_size_value = None
    else:
        batch_size = batch_size_value = input_tensor.get_shape().as_list()[0]
    state_phs = nest.map_structure(
        lambda size: tf.compat.v1.placeholder_with_default(
            tf.zeros([batch_size, size]), [batch_size_value, size]
        ),
        rnn.cell.state_size,
    )
    state_keras_inputs = nest.map_structure(
        lambda ph: tf.keras.Input(tensor=ph), state_phs
    )
    input_layer = tf.keras.Input(tensor=input_tensor)

    out, *states = rnn(input_layer, initial_state=state_keras_inputs, mask=input_mask)

    state_phs_flat = nest.flatten(state_phs)
    states_flat = nest.flatten(states)

    return out, states_flat, state_phs_flat


class StackedLSTMCells(tf.keras.layers.StackedRNNCells):

    _flatten_state_size = True

    @property
    def state_size(self):
        state_size = super(StackedLSTMCells, self).state_size
        if self._flatten_state_size:
            state_size = nest.flatten(state_size)
        return state_size

    def call(self, inputs, states, constants=None, **kwargs):
        flatten_state_size = self._flatten_state_size
        self._flatten_state_size = False
        output, new_states = super(StackedLSTMCells, self).call(
            inputs, states, constants=None, **kwargs
        )
        self._flatten_state_size = flatten_state_size

        new_states = nest.flatten(new_states)
        return output, new_states


def build_input_mask(sequence_lengths_tensor):
    batch_size = tf.shape(sequence_lengths_tensor)[0]
    max_seq_len = tf.reduce_max(sequence_lengths_tensor)
    step_indices = tf.range(max_seq_len)
    step_indices = tf.tile(tf.expand_dims(step_indices, 0), [batch_size, 1])
    mask = tf.less(step_indices, tf.expand_dims(sequence_lengths_tensor, 1))
    mask = tf.expand_dims(mask, axis=2)
    return mask
