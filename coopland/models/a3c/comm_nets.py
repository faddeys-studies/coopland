import tensorflow as tf
import numpy as np
import dataclasses
import os
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation
from coopland.models.a3c import config_lib
from coopland import tf_utils


def create(hparams: config_lib.AgentModelHParams, cell: tf.keras.layers.Layer):
    if hparams.comm_type in (None, "signal_as_feature"):
        return CommCellSignalAsFeature(cell, hparams.comm_units[0])
    elif hparams.comm_type == "state_avg_reader":
        return CommCellStateAvgReader(cell)
    elif hparams.comm_type == "inner_rnn":
        return CommCellInnerRNN(cell, hparams.comm_units)
    raise ValueError(hparams.comm_type)


class BaseCommCell(tf.keras.layers.Layer):
    signal_size = 0

    def __init__(self, wrapped_cell, signal_size):
        super(BaseCommCell, self).__init__()
        self.rnn_cell: tf.keras.layers.StackedRNNCells = wrapped_cell
        self.signal_size = signal_size

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
            *(v for l in self._get_additional_layers() for v in l.trainable_weights),
        ]

    def _get_additional_layers(self):
        return []

    def get_visible_ids(self, visible_other_agents):
        raise NotImplementedError


class CommCellSignalAsFeature(BaseCommCell):
    def __init__(self, rnn_cell, signal_size):
        super(CommCellSignalAsFeature, self).__init__(rnn_cell, signal_size)
        self.signal_generator = tf.keras.layers.Dense(
            signal_size, activation=tf.nn.leaky_relu
        )
        self.directions_list = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.directions_list)}

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, signals = nest.pack_sequence_as(self._state_structure, states)

        assert signals.get_shape().as_list()[1] == self.signal_size

        signals_sets, _ = _gather_present(signals, present_indices)
        signal_features = tf.reshape(
            signals_sets,
            [
                tf_utils.get_shape_static_or_dynamic(signals_sets)[0],
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
        input_shape = tuple(input_shape)
        self.rnn_cell.build(
            input_shape[:-1] + (input_shape[-1] + 4 * self.signal_size,)
        )
        self.signal_generator.build(input_shape[:-1] + (self.rnn_cell.output_size,))
        self.built = True

    def _get_additional_layers(self):
        return [self.signal_generator]


class CommCellStateAvgReader(BaseCommCell):
    def __init__(self, rnn_cell):
        super(CommCellStateAvgReader, self).__init__(rnn_cell, 0)

    def call(self, inputs, rnn_states=None, **kwargs):
        assert rnn_states is not None
        inputs, present_indices = inputs
        orig_states = rnn_states
        rnn_states = nest.flatten(rnn_states)

        readed_states = rnn_states[-1]
        readed_states, mask = _gather_present(readed_states, present_indices)
        mask_float = tf.expand_dims(tf.cast(mask, tf.float32), 2)
        features = tf.reduce_sum(mask_float * readed_states, axis=1) / (
            1e-5 + tf.reduce_sum(mask_float, axis=1)
        )

        full_input = tf.concat([inputs, features], axis=1)

        features, *rnn_states_after = self.rnn_cell.call(full_input, rnn_states)
        rnn_states_after = nest.pack_sequence_as(
            orig_states, nest.flatten(rnn_states_after)
        )
        return features, rnn_states_after

    def get_visible_ids(self, visible_other_agents):
        return [ag_id for ag_id, direction, dist in visible_other_agents]

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        wrapped_state_sizes = nest.flatten(self.rnn_cell.state_size)
        self.rnn_cell.build(
            input_shape[:-1] + (input_shape[-1] + wrapped_state_sizes[-1],)
        )
        self.built = True

    def _get_additional_layers(self):
        return []

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size


class CommCellInnerRNN(BaseCommCell):
    def __init__(self, rnn_cell, units):
        super(CommCellInnerRNN, self).__init__(rnn_cell, units[-1])
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.LSTMCell(n) for n in units]
            ),
            return_state=True,
        )
        self.directions_list = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.directions_list)}

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, comm_states, signals = nest.pack_sequence_as(
            self._state_structure, states
        )

        # run main recurrent unit:
        features, rnn_states_after = self.rnn_cell.call(inputs, rnn_states)

        # unpack info about who and from where is we are communicating
        vis_dists = tf.cast(present_indices[:, 1::3], tf.float32)
        vis_dirs = present_indices[:, 2::3]
        present_indices = present_indices[:, 0::3]
        n_agents, max_others = tf_utils.get_shape_static_or_dynamic(present_indices)
        vis_dirs_onehot = tf.cast(tf.one_hot(vis_dirs, 4), tf.float32)

        # add additional comm step to share info with self
        present_indices = tf.concat(
            [
                present_indices,
                tf.expand_dims(tf.range(n_agents, dtype=tf.int32), axis=1),
            ],
            axis=1,
        )
        vis_dists = tf.pad(vis_dists, [(0, 0), (0, 1)])
        vis_dirs_onehot = tf.pad(vis_dirs_onehot, [(0, 0), (0, 1), (0, 0)])

        # put all comm net inputs together:
        signal_inputs, signals_mask = _gather_present(signals, present_indices)
        signal_inputs = tf.concat(
            [
                signal_inputs,
                vis_dirs_onehot,
                tf.expand_dims(vis_dists, 2),
                tf.tile(tf.expand_dims(features, 1), [1, max_others+1, 1]),
            ],
            axis=2,
        )

        # finally call communication steps:
        out_signals, *new_comm_states = self.comm_rnn.call(
            signal_inputs, signals_mask, initial_state=comm_states, constants=()
        )

        # pack for output:
        states_after = rnn_states_after, new_comm_states, out_signals
        full_output = tf.concat([out_signals, out_signals, features], axis=1)
        return full_output, tuple(nest.flatten(states_after))

    def get_visible_ids(self, visible_other_agents):
        result = []
        for ag_id, direction, dist in visible_other_agents:
            result.extend([ag_id, dist, self.directions_to_i[direction]])
        # if not result:
        #     result = [-1, -1, -1]
        return result

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.rnn_cell.build(input_shape)
        self.comm_rnn.build(
            (input_shape[0], None, self.signal_size + 5 + self.rnn_cell.output_size)
        )
        self.built = True

    def _get_additional_layers(self):
        return [self.comm_rnn]

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.comm_rnn.cell.state_size, self.signal_size

    @property
    def output_size(self):
        return self.signal_size * 2 + self.rnn_cell.output_size


def _gather_present(vectors, present_indices):
    vectors = tf.pad(vectors, [(1, 0), (0, 0)])
    return tf.gather(vectors, present_indices + 1), tf.greater_equal(present_indices, 0)
