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

        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units) for units in self.hparams.rnn_units]
        )
        if self.hparams.use_communication:
            comm_net = CommCell(self.hparams.comm_units, self.hparams.rnn_units[-1])
            cell = CommCellWrapper(cell, comm_net)
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
        states = {}
        signals = {}
        times = {}

        def agent_fn(*observation):
            agent_id = observation[0]
            t = times.get(agent_id, 0) + 1
            input_data, metadata = self.encode_observation(*observation)
            _run_tensors = (actor_probabilities_t, critic_t), new_states_t

            prev_states = states.get((agent_id, t - 1), [])
            feed = {input_ph: [input_data]}
            feed.update(zip(prev_states_phs, prev_states))
            if self.hparams.use_communication:
                visible_other_agents = observation[3]
                other_agent_ids = [ag_id for ag_id, _, _ in visible_other_agents]
                others_signals = []
                for ag_id in other_agent_ids:
                    others_signal = signals.get((ag_id, t - 1), None)
                    if others_signal is None:
                        others_signal = np.zeros([1, *signals_shape])
                    others_signals.append(others_signal)
                if others_signals:
                    others_signals = np.concatenate(others_signals, axis=0)
                else:
                    others_signals = np.zeros([0, *signals_shape])
                feed[others_signals_ph] = others_signals
                feed[present_indices] = [[np.arange(others_signals.shape[0])]]
                _run_tensors = _run_tensors, out_signal_t

            _out_values = session.run(_run_tensors, feed)
            if self.hparams.use_communication:
                _out_values, out_signal = _out_values
                signals[agent_id, t] = out_signal[:, 0]
                signals.pop((agent_id, t - 2), None)
            output_data, new_states = _out_values
            states[agent_id, t] = new_states
            times[agent_id] = t
            states.pop((agent_id, t - 2), None)
            move = self.decode_nn_output(output_data, metadata, greed_choice_prob)
            return move

        def init_before_game():
            states.clear()
            times.clear()
            signals.clear()

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        input_ph = tf.compat.v1.placeholder(tf.float32, [1, None, self.input_data_size])
        if self.hparams.use_communication:
            assert isinstance(agent_instance.rnn.cell, CommCellWrapper)
            others_signals_ph = (
                agent_instance.rnn.cell.comm_net.get_placeholder_for_others_signals()
            )
            signals_shape = others_signals_ph.get_shape().as_list()[1:]
            present_indices = tf.compat.v1.placeholder(tf.int32, [1, 1, None])
        else:
            others_signals_ph = present_indices = None
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_states_t,
            prev_states_phs,
            out_signal_t,
        ] = agent_instance.call(
            input_ph,
            const_others_signals_tensor=others_signals_ph,
            present_indices=present_indices,
        )
        del actor_logits_t

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
        const_others_signals_tensor=None,
        present_indices: "[N_batch_agents time max_other_agents]" = None,
    ):
        if input_mask is None:
            if sequence_lengths_tensor is not None:
                input_mask = build_input_mask(sequence_lengths_tensor)

        if self.model_hparams.use_communication:
            assert isinstance(self.rnn.cell, CommCellWrapper)
            self.rnn.cell.const_others_signals = const_others_signals_tensor
            input_tensor = input_tensor, present_indices
        signals, states_after, states_before_phs = _call_rnn(
            self.rnn, input_tensor, input_mask
        )
        actor_logits = self.actor_head(signals)
        critic_value = self.critic_head(signals)[:, :, 0]
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


class CommCellWrapper(tf.keras.layers.Layer):
    def __init__(self, rnn_cell, comm_cell):
        super(CommCellWrapper, self).__init__()
        self.rnn_cell: tf.keras.layers.StackedRNNCells = rnn_cell
        self.comm_net: CommCell = comm_cell
        self.const_others_signals = None

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, comm_states = nest.pack_sequence_as(self._state_structure, states)

        features, rnn_states_after = self.rnn_cell.call(inputs, rnn_states)
        comm_signals, comm_state_after = self.comm_net.call(
            features=features,
            comm_states=comm_states,
            present_indices=present_indices,
            others_signals=self.const_others_signals,
        )
        states_after = rnn_states_after, comm_state_after
        return comm_signals, tuple(nest.flatten(states_after))

    def build(self, input_shape):
        self.rnn_cell.build(input_shape)
        self.comm_net.build((input_shape[0], self.rnn_cell.output_size))
        self.built = True

    @property
    def output_size(self):
        return self.comm_net.output_size

    @property
    def state_size(self):
        return tuple(nest.flatten(self._state_structure))

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.comm_net.state_size

    @property
    def trainable_weights(self):
        return [*self.rnn_cell.trainable_weights, *self.comm_net.trainable_weights]


class CommCell(tf.keras.layers.Layer):
    def __init__(self, units, features_size):
        super(CommCell, self).__init__()
        self._units = units
        self._features_size = features_size
        self._out_dim = self._units[-1]
        self._comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(units) for units in self._units]
            ),
            return_state=True,
        )

    def call(
        self,
        features,
        comm_states=None,
        present_indices=None,
        others_signals=None,
        **kwargs,
    ):
        assert comm_states is not None
        assert present_indices is not None
        signals, inner_rnn_states = comm_states
        assert signals.get_shape().as_list()[1] == self._out_dim

        others_signals = others_signals if others_signals is not None else signals
        others_signals_pad = tf.pad(others_signals, [(1, 0), (0, 0)])
        others_signals_seqs = tf.gather(others_signals_pad, present_indices + 1)
        signals_mask = tf.greater_equal(present_indices, 0)

        all_signals_seq = tf.concat(
            [tf.expand_dims(signals, 1), others_signals_seqs], 1
        )
        signals_mask = tf.pad(signals_mask, [(0, 0), (1, 0)], constant_values=True)

        n_comm_steps = tf.shape(all_signals_seq)[1]
        inner_rnn_full_input = tf.concat(
            [
                all_signals_seq,
                tf.tile(tf.expand_dims(features, 1), [1, n_comm_steps, 1]),
            ],
            axis=2,
        )

        out_signal, *new_inner_rnn_states = self._comm_rnn.call(
            inner_rnn_full_input,
            signals_mask,
            initial_state=inner_rnn_states,
            constants=(),
        )
        return out_signal, (out_signal, new_inner_rnn_states)

    def build(self, input_shape):
        assert input_shape[1] == self._features_size
        self._comm_rnn.build((input_shape[0], None, self._out_dim + input_shape[1]))
        self.built = True

    def get_placeholder_for_others_signals(self):
        return tf.compat.v1.placeholder(tf.float32, [None, self._out_dim])

    @property
    def output_size(self):
        return self._out_dim

    @property
    def state_size(self):
        return self._out_dim, self._comm_rnn.cell.state_size

    @property
    def trainable_weights(self):
        return self._comm_rnn.trainable_weights
