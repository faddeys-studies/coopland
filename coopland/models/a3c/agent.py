import tensorflow as tf
import numpy as np
import dataclasses
import os
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation


class AgentModel:

    INPUT_DATA_SIZE = 4 + 8 + 5  # visibility + corners + exit

    def __init__(self):
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

        vector = np.array(result)
        assert vector.shape == (self.INPUT_DATA_SIZE,)
        assert vector.dtype == np.int
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

        actor_rnn = tf.keras.layers.RNN(
            StackedLSTMCells(
                [
                    tf.keras.layers.LSTMCell(100),
                    tf.keras.layers.LSTMCell(100),
                    tf.keras.layers.LSTMCell(100),
                    tf.keras.layers.LSTMCell(100),
                ]
            ),
            return_state=True,
            return_sequences=True,
            name=name_prefix + "Actor/RNN",
        )
        actor_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(4), name=name_prefix + "Actor/Head"
        )
        critic_rnn = tf.keras.layers.RNN(
            StackedLSTMCells(
                [
                    tf.keras.layers.LSTMCell(100),
                    tf.keras.layers.LSTMCell(100),
                    tf.keras.layers.LSTMCell(100),
                    tf.keras.layers.LSTMCell(100),
                ]
            ),
            return_state=True,
            return_sequences=True,
            name=name_prefix + "Critic/RNN",
        )
        critic_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1), name=name_prefix + "Critic/Head"
        )

        actor_rnn.build((None, None, self.INPUT_DATA_SIZE))
        actor_head.build((None, None, actor_rnn.cell.cells[-1].units))
        critic_rnn.build((None, None, self.INPUT_DATA_SIZE))
        critic_head.build((None, None, critic_rnn.cell.cells[-1].units))

        saver = tf.train.Saver(
            [
                *actor_rnn.trainable_variables,
                *critic_rnn.trainable_variables,
                *actor_head.trainable_variables,
                *critic_head.trainable_variables,
            ]
        )
        return AgentInstance(actor_rnn, critic_rnn, actor_head, critic_head, saver)

    def create_agent_fn(
        self, agent_instance: "AgentInstance", session, greed_choice_prob=None
    ):
        actor_states = []
        critic_states = []

        def agent_fn(*observation):
            input_data, metadata = self.encode_observation(*observation)

            feed = {input_ph: [input_data]}
            feed.update(zip(prev_actor_states_phs, actor_states))
            feed.update(zip(prev_critic_states_phs, critic_states))

            output_data, (new_actor_states, new_critic_states) = session.run(
                (
                    (actor_probabilities_t, critic_t),
                    (new_actor_states_t, new_critic_states_t),
                ),
                feed,
            )
            actor_states[:] = new_actor_states
            critic_states[:] = new_critic_states
            move = self.decode_nn_output(output_data, metadata, greed_choice_prob)
            return move

        def init_before_game():
            actor_states.clear()
            critic_states.clear()

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        input_ph = agent_instance.make_input_ph(1)
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_actor_states_t,
            new_critic_states_t,
            prev_actor_states_phs,
            prev_critic_states_phs,
        ] = agent_instance.call(input_ph)
        del actor_logits_t

        return agent_fn


@dataclasses.dataclass
class AgentInstance:
    actor_rnn: "tf.keras.layers.RNN"
    critic_rnn: "tf.keras.layers.RNN"
    actor_head: "tf.keras.layers.Layer"
    critic_head: "tf.keras.layers.Layer"
    saver: "tf.train.Saver"

    def call(self, input_tensor):

        actor_logits, actor_after_states, actor_before_states_phs = _call_stateful_rnn(
            self.actor_rnn, self.actor_head, input_tensor
        )
        critic_value, critic_after_states, critic_before_states_phs = _call_stateful_rnn(
            self.critic_rnn, self.critic_head, input_tensor
        )
        critic_value = critic_value[:, :, 0]
        actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)

        return (
            actor_logits,
            actor_probabilities,
            critic_value,
            actor_after_states,
            critic_after_states,
            actor_before_states_phs,
            critic_before_states_phs,
        )

    def get_variables(self):
        layers = self.actor_rnn, self.actor_head, self.critic_rnn, self.critic_head
        return [v for layer in layers for v in layer.trainable_variables]

    @property
    def actor_trainable_variables(self):
        return self.actor_rnn.trainable_variables + self.actor_head.trainable_variables

    @property
    def critic_trainable_variables(self):
        return (
            self.critic_rnn.trainable_variables + self.critic_head.trainable_variables
        )

    def save_variables(self, session, directory, step):
        checkpoint_path = os.path.join(directory, "model.ckpt")
        self.saver.save(
            session, checkpoint_path, global_step=step, write_meta_graph=False
        )
        with open(os.path.join(directory, "step.txt"), "w") as f:
            f.write(str(step))

    def load_variables(self, session, directory):
        save_path = tf.train.latest_checkpoint(directory)
        self.saver.restore(session, save_path)
        with open(os.path.join(directory, "step.txt"), "r") as f:
            return int(f.read().strip())

    @staticmethod
    def make_input_ph(batch_size=1):
        return tf.compat.v1.placeholder(
            tf.float32, [batch_size, None, AgentModel.INPUT_DATA_SIZE]
        )


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


def _call_stateful_rnn(rnn, post_layer, input_tensor):
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

    features, *states = rnn(input_layer, initial_state=state_keras_inputs)
    out = post_layer(features)

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
        try:
            return super(StackedLSTMCells, self).call(
                inputs, states, constants=None, **kwargs
            )
        finally:
            self._flatten_state_size = flatten_state_size
