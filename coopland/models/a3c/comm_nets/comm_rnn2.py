import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseCommCell
from coopland.models.a3c import util


class CommRNN2(BaseCommCell):
    def __init__(self, rnn_cell, units, use_gru, use_bidir, can_see_others):
        self._can_see_others = can_see_others
        super(CommRNN2, self).__init__(rnn_cell, units[-1])
        cell_type = tf.keras.layers.GRUCell if use_gru else tf.keras.layers.LSTMCell
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([cell_type(n) for n in units])
        )
        self._comm_out_size = self.comm_rnn.cell.output_size
        self._env_out_size = self.rnn_cell.output_size
        if use_bidir:
            self.comm_rnn = tf.keras.layers.Bidirectional(self.comm_rnn, "sum")

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        rnn_states, signals = states

        env_features, rnn_states_after = self.rnn_cell.call(inputs, rnn_states)

        signal_sets, signals_mask = util.gather_present(
            signals, comm_indices, prepend_own=True
        )
        if self._can_see_others:
            signal_sets = util.add_visible_agents_to_each_timestep(
                signal_sets, comm_directions, comm_distances, prepend_own=True
            )

        n_comm_steps = tf.shape(signal_sets)[1]
        env_features_broadcasted = tf.tile(
            tf.expand_dims(env_features, 1),
            [1, n_comm_steps, 1]
        )

        full_comm_inputs = tf.concat([signal_sets, env_features_broadcasted], axis=2)

        out_features = self.comm_rnn.call(full_comm_inputs, signals_mask)

        return out_features, (rnn_states_after, out_features)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.rnn_cell.build(input_shape)
        comm_in_size = self.signal_size
        if self._can_see_others:
            comm_in_size += 5
        comm_in_size += self._env_out_size
        self.comm_rnn.build((input_shape[0], None, comm_in_size))
        self.built = True

    def _get_additional_layers(self):
        return [self.comm_rnn]

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.signal_size

    @property
    def output_size(self):
        return self.signal_size
