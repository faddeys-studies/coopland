import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseCommCell
from coopland.models.a3c import util


class CommRNN1(BaseCommCell):
    def __init__(self, rnn_cell, units, use_gru, use_bidir, can_see_others, version):
        super(CommRNN1, self).__init__(rnn_cell, 0)
        cell_type = tf.keras.layers.GRUCell if use_gru else tf.keras.layers.LSTMCell
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([cell_type(n) for n in units])
        )
        self.signal_size = units[-1]
        self._can_see_others = can_see_others
        self.version = version
        self._comm_out_size = self.comm_rnn.cell.output_size
        if use_bidir:
            self.comm_rnn = tf.keras.layers.Bidirectional(self.comm_rnn, "sum")

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        if self.version == 1:
            signals = states[-1][0]
        else:
            states, signals = states
        signals, mask = util.gather_present(signals, comm_indices, prepend_own=True)
        if self._can_see_others:
            signals = util.add_visible_agents_to_each_timestep(
                signals, comm_directions, comm_distances, prepend_own=True
            )
        comm_result = self.comm_rnn.call(signals, mask)

        full_input = tf.concat([inputs, comm_result], axis=1)

        features, *states_after = self.rnn_cell.call(full_input, states)
        if self.version == 2:
            states_after = states_after, comm_result
        return features, states_after

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        comm_in_size = self.get_signal(self._state_structure)
        if self._can_see_others:
            comm_in_size += 5
        self.comm_rnn.build((input_shape[0], None, comm_in_size))
        self.rnn_cell.build(input_shape[:-1] + (self._comm_out_size + input_shape[-1],))
        self.built = True

    def _get_additional_layers(self):
        return [self.comm_rnn]

    @property
    def _state_structure(self):
        if self.version == 1:
            return self.rnn_cell.state_size
        else:
            return self.rnn_cell.state_size, self.signal_size

    def get_signal(self, states):
        states = nest.pack_sequence_as(self._state_structure, nest.flatten(states))
        if self.version == 1:
            return states[-1][0]
        else:
            states, signals = states
            return signals
