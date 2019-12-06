import tensorflow as tf
import numpy as np
import dataclasses
import os
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation
from coopland.models.a3c import config_lib, comm_nets, util


class AgentModel:
    def __init__(self, hparams: config_lib.AgentModelHParams):
        self.hparams = hparams
        self.input_data_size = 4 + 8 + 5  # visibility + corners + exit
        self.directions_list = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.directions_list)}

    def encode_observation(
        self, agent_id, visibility, corners, visible_other_agents, visible_exit
    ):
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
        assert vector.shape == (self.input_data_size,)
        return vector

    def build_layers(self, name=None):
        if name:
            name_prefix = name + "_"
        else:
            name_prefix = ""

        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units) for units in self.hparams.rnn_units]
        )
        if self.hparams.comm is not None:
            cell = comm_nets.create(self.hparams, cell)
        rnn = tf.keras.layers.RNN(
            cell, return_state=True, return_sequences=True, name=name_prefix + "RNN"
        )
        actor_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(4), name=name_prefix + "Actor/Head"
        )
        critic_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1), name=name_prefix + "Critic/Head"
        )

        rnn.build((None, None, self.input_data_size))
        actor_head.build((None, None, cell.output_size))
        critic_head.build((None, None, cell.output_size))

        saver = tf.train.Saver(
            [
                *rnn.trainable_variables,
                *actor_head.trainable_variables,
                *critic_head.trainable_variables,
            ]
        )
        return AgentInstance(self.hparams, rnn, actor_head, critic_head, saver)

    def create_agent_fn(
        self, agent_instance: "AgentInstance", session, greed_choice_prob=None
    ):
        n_agents = 0
        observations = []
        states = []
        result_dict = {}

        def compute_fn():
            input_batch = np.zeros([n_agents, self.input_data_size])
            comm_data = util.CommData(n_agents, 1)
            for observation in observations:
                if observation is None:
                    continue
                agent_id = observation[0]
                input_batch[agent_id] = self.encode_observation(*observation)
                comm_data.add_observation(observation)
            comm_ids_batch, comm_dirs_batch, comm_dist_batch = comm_data.build_batch()
            feed = {
                input_ph: np.expand_dims(input_batch, axis=1),
                comm_indices_ph: comm_ids_batch,
                comm_directions_ph: comm_dirs_batch,
                comm_distances_ph: comm_dist_batch,
                input_mask_ph: np.array(
                    [[observation is not None] for observation in observations]
                ),
            }
            feed.update(zip(prev_states_phs, states))

            tensors = actor_probabilities_t, critic_t, new_states_t
            actor_probs, critic, new_states = session.run(tensors, feed)

            observations[:] = [None] * n_agents
            for state_buf, new_state in zip(states, new_states):
                state_buf[...] = new_state
            nonlocal result_dict
            result_dict = {}
            return {
                "probabilities": actor_probs[:, 0, :],
                "critic_value": critic[:, 0],
                "inputs": input_batch,
            }

        def agent_fn(*observation):
            agent_id = observation[0]
            observations[agent_id] = observation
            return Move(
                result_dict, compute_fn, agent_id, observation, greed_choice_prob
            )

        def init_before_game(n_agents_):
            nonlocal states, observations, n_agents
            n_agents = n_agents_
            observations = [None] * n_agents_
            states = [
                np.zeros([n_agents_, st_ph.get_shape().as_list()[1]])
                for st_ph in prev_states_phs
            ]

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        input_ph = tf.compat.v1.placeholder(tf.float32, [None, 1, self.input_data_size])
        comm_indices_ph = tf.compat.v1.placeholder(tf.int32, [None, 1, None])
        comm_directions_ph = tf.compat.v1.placeholder(tf.int32, [None, 1, None])
        comm_distances_ph = tf.compat.v1.placeholder(tf.float32, [None, 1, None])
        input_mask_ph = tf.compat.v1.placeholder(tf.bool, [None, 1])
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_states_t,
            prev_states_phs,
            out_signal_t,
        ] = agent_instance.call(
            input_ph,
            input_mask=input_mask_ph,
            comm_indices=comm_indices_ph,
            comm_directions=comm_directions_ph,
            comm_distances=comm_distances_ph,
        )
        del actor_logits_t
        del out_signal_t

        return agent_fn


@dataclasses.dataclass
class AgentInstance:
    model_hparams: config_lib.AgentModelHParams
    rnn: "tf.keras.layers.RNN"
    actor_head: "tf.keras.layers.Layer"
    critic_head: "tf.keras.layers.Layer"
    saver: "tf.train.Saver"

    def call(
        self,
        input_tensor,
        sequence_lengths_tensor=None,
        input_mask=None,
        *,
        comm_indices: "[N_batch_agents time max_other_agents]",
        comm_distances: "[N_batch_agents time max_other_agents]",
        comm_directions: "[N_batch_agents time max_other_agents]",
    ):
        if input_mask is None:
            if sequence_lengths_tensor is not None:
                input_mask = build_input_mask(sequence_lengths_tensor)

        if self.model_hparams.comm:
            assert isinstance(self.rnn.cell, comm_nets.BaseCommCell)
            input_tensor = input_tensor, comm_indices, comm_directions, comm_distances
        features, states_after, states_before_phs = _call_rnn(
            self.rnn, input_tensor, input_mask
        )
        if self.model_hparams.comm:
            signals = self.rnn.cell.get_signal(states_after)
        else:
            signals = None
        actor_logits = self.actor_head(features)
        critic_value = self.critic_head(features)[:, :, 0]
        actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)

        return (
            actor_logits,
            actor_probabilities,
            critic_value,
            states_after,
            states_before_phs,
            signals,
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


class Move:
    def __init__(
        self, results_dict, compute_fn, agent_id, observation, greed_choice_prob
    ):
        self._results_dict = results_dict
        self._compute_fn = compute_fn
        self._agent_id = agent_id
        self._direction_idx = None
        self._greed_choice_prob = greed_choice_prob
        self.observation: Observation = observation
        self.debug_text: str = ""

    @property
    def probabilities(self):
        if not self._results_dict:
            self._results_dict.update(self._compute_fn())
        return self._results_dict["probabilities"][self._agent_id]

    @property
    def critic_value(self):
        if not self._results_dict:
            self._results_dict.update(self._compute_fn())
        return self._results_dict["critic_value"][self._agent_id]

    @property
    def direction_idx(self):
        if self._direction_idx is None:
            probs = self.probabilities
            if (
                self._greed_choice_prob is not None
                and np.random.random() < self._greed_choice_prob
            ):
                direction_i = np.argmax(probs)
            else:
                direction_i = np.random.choice(range(len(probs)), p=probs)
            self._direction_idx = int(direction_i)
        return self._direction_idx

    @property
    def direction(self):
        return _directions[self.direction_idx]

    @property
    def input_vector(self):
        if not self._results_dict:
            self._results_dict.update(self._compute_fn())
        return self._results_dict["inputs"][self._agent_id]

    def __repr__(self):
        return f"<{self.direction} V={self.critic_value:3f} A={self.probabilities}>"


def _call_rnn(rnn: tf.keras.layers.RNN, input_tensor, input_mask=None):
    if isinstance(input_tensor, (list, tuple)):
        some_input = input_tensor[0]
    else:
        some_input = input_tensor
    if some_input.get_shape().as_list()[0] is None:
        batch_size = tf.shape(some_input)[0]
        batch_size_value = None
    else:
        batch_size = batch_size_value = some_input.get_shape().as_list()[0]
    state_phs = nest.map_structure(
        lambda size: tf.compat.v1.placeholder_with_default(
            tf.zeros([batch_size, size]), [batch_size_value, size]
        ),
        rnn.cell.state_size,
    )
    state_keras_inputs = nest.map_structure(
        lambda ph: tf.keras.Input(tensor=ph), state_phs
    )
    keras_inputs = nest.map_structure(lambda t: tf.keras.Input(tensor=t), input_tensor)

    out, *states = rnn.call(
        keras_inputs, initial_state=state_keras_inputs, mask=input_mask, constants=()
    )

    state_phs_flat = nest.flatten(state_phs)
    states_flat = nest.flatten(states)

    return out, states_flat, state_phs_flat


def build_input_mask(sequence_lengths_tensor):
    batch_size = tf.shape(sequence_lengths_tensor)[0]
    max_seq_len = tf.reduce_max(sequence_lengths_tensor)
    step_indices = tf.range(max_seq_len)
    step_indices = tf.tile(tf.expand_dims(step_indices, 0), [batch_size, 1])
    mask = tf.less(step_indices, tf.expand_dims(sequence_lengths_tensor, 1))
    mask = tf.expand_dims(mask, axis=2)
    return mask


_directions = Direction.list_clockwise()
