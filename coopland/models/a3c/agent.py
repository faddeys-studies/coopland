import tensorflow as tf
import numpy as np
import dataclasses
import os
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation
from coopland.models.a3c import config_lib
from coopland import tf_utils


class AgentModel:
    def __init__(self, hparams: config_lib.AgentModelHParams):
        self.hparams = hparams
        self.input_data_size = 4 + 8 + 5  # visibility + corners + exit
        if self.hparams.use_visible_agents:
            # +4 distances to other agents
            self.input_data_size += 4 * (hparams.max_agents - 1)
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
        return vector

    def build_layers(self, name=None):
        if name:
            name_prefix = name + "_"
        else:
            name_prefix = ""

        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units) for units in self.hparams.rnn_units]
        )
        if self.hparams.use_communication:
            cell = CommCell(cell, self.hparams.comm_units[0])
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
        actor_head.build((None, None, rnn.cell.output_size))
        critic_head.build((None, None, rnn.cell.output_size))

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
            present_ids_batch = [[] for _ in range(n_agents)]
            for observation in observations:
                if observation is None:
                    continue
                agent_id = observation[0]
                input_batch[agent_id] = self.encode_observation(*observation)
                present_ids_batch[agent_id] = agent_instance.get_visible_ids(observation[3])
            present_ids_batch = tf.keras.preprocessing.sequence.pad_sequences(
                present_ids_batch, dtype=int, padding="post", value=-1
            )
            feed = {
                input_ph: np.expand_dims(input_batch, axis=1),
                present_indices_ph: np.expand_dims(present_ids_batch, axis=1),
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
        present_indices_ph = tf.compat.v1.placeholder(tf.int32, [None, 1, None])
        input_mask_ph = tf.compat.v1.placeholder(tf.bool, [None, 1])
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_states_t,
            prev_states_phs,
            out_signal_t,
        ] = agent_instance.call(
            input_ph, input_mask=input_mask_ph, present_indices=present_indices_ph
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
        present_indices: "[N_batch_agents time max_other_agents]" = None,
    ):
        if input_mask is None:
            if sequence_lengths_tensor is not None:
                input_mask = build_input_mask(sequence_lengths_tensor)
        assert present_indices is not None

        if self.model_hparams.use_communication:
            assert isinstance(self.rnn.cell, BaseCommCell)
            input_tensor = input_tensor, present_indices
        features, states_after, states_before_phs = _call_rnn(
            self.rnn, input_tensor, input_mask
        )
        if self.model_hparams.use_communication:
            signal_size = self.rnn.cell.signal_size
            signals = features[:, :signal_size]
            features = features[:, signal_size:]
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

    def get_visible_ids(self, visible_other_agents):
        if not self.model_hparams.use_communication:
            return []
        return self.rnn.cell.get_visible_ids(visible_other_agents)

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


class BaseCommCell:
    signal_size = 0

    def get_visible_ids(self, visible_other_agents):
        raise NotImplementedError


class CommCell(BaseCommCell, tf.keras.layers.Layer):
    def __init__(self, rnn_cell, signal_size):
        super(CommCell, self).__init__()
        self.rnn_cell: tf.keras.layers.StackedRNNCells = rnn_cell
        self.signal_generator = tf.keras.layers.Dense(
            signal_size, activation=tf.nn.leaky_relu
        )
        self.signal_size = signal_size

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, signals = nest.pack_sequence_as(self._state_structure, states)

        assert signals.get_shape().as_list()[1] == self.signal_size
        assert present_indices.get_shape().as_list()[1] == 4

        signals = signals
        signals_pad = tf.pad(signals, [(1, 0), (0, 0)])
        signals_sets = tf.gather(signals_pad, present_indices + 1)
        signal_features = tf.transpose(signals_sets, [1, 0, 2])
        assert signal_features.get_shape().as_list() == [
            signals.get_shape().as_list()[0],
            4,
            self.signal_size,
        ]
        signal_features = tf.reshape(
            signal_features,
            [
                tf_utils.get_shape_static_or_dynamic(signal_features)[0],
                4 * self.signal_size,
            ],
        )

        full_input = tf.concat([inputs, signal_features], axis=1)

        features, rnn_states_after = self.rnn_cell.call(full_input, rnn_states)
        new_own_signal = self.signal_generator.call(features)
        states_after = rnn_states_after, new_own_signal
        full_output = tf.concat([new_own_signal, features], axis=1)
        return full_output, tuple(nest.flatten(states_after))

    def get_visible_ids(self, visible_other_agents):
        present_indices = [-1] * 4
        present_distances = [None] * 4
        for ag_id, direction, dist in visible_other_agents:
            i = self.directions_to_i[direction]
            if present_distances[i] is None or dist < present_distances[i]:
                present_indices[i] = ag_id
        return present_indices

    def build(self, input_shape):
        input_shape = list(input_shape)
        self.rnn_cell.build(input_shape[:-1] + [input_shape[-1] + self.signal_size])
        self.signal_generator.build(input_shape[:-1] + [self.rnn_cell.output_size])
        self.built = True

    @property
    def output_size(self):
        return self.signal_size + self.rnn_cell.output_size

    @property
    def state_size(self):
        return tuple(nest.flatten(self._state_structure))

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.signal_size

    @property
    def trainable_weights(self):
        return [
            *self.rnn_cell.trainable_weights,
            *self.signal_generator.trainable_weights,
        ]


_directions = Direction.list_clockwise()
