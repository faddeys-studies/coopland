import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseCommCell
from . import _util


class CommCellInnerRNN(BaseCommCell):
    def __init__(
        self, rnn_cell, units, signal_dropout_rate, version, use_gru, use_bidir
    ):
        signal_size = 23
        assert version in (1, 2)
        self.version = version
        self._outputs_size = 31
        self._signal_dropout_rate = signal_dropout_rate
        super(CommCellInnerRNN, self).__init__(rnn_cell, signal_size)
        cell_type = tf.keras.layers.GRUCell if use_gru else tf.keras.layers.LSTMCell
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([cell_type(n) for n in units])
        )
        self._comm_out_size = self.comm_rnn.cell.output_size
        if use_bidir:
            self.comm_rnn = tf.keras.layers.Bidirectional(self.comm_rnn, "sum")
        self.final_layer = tf.keras.layers.Dense(
            self._outputs_size + self.signal_size, activation=tf.nn.leaky_relu
        )

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, signals = nest.pack_sequence_as(self._state_structure, states)
        if self._signal_dropout_rate is not None:
            signals = tf.nn.dropout(signals, rate=self._signal_dropout_rate)

        # unpack info about who and from where is we are communicating
        vis_dists = tf.cast(present_indices[:, 1::3], tf.float32)
        vis_dirs = present_indices[:, 2::3]
        present_indices = present_indices[:, 0::3]
        vis_dirs_onehot = tf.cast(tf.one_hot(vis_dirs, 4), tf.float32)

        # put all comm net inputs together:
        signal_inputs, signals_mask = _util.gather_present(signals, present_indices)
        # signal_inputs = tf.concat(
        #     [signal_inputs, vis_dirs_onehot, tf.expand_dims(vis_dists, 2)], axis=2
        # )

        # finally call communication steps:
        comm_features = self.comm_rnn.call(signal_inputs, signals_mask)

        # run main recurrent unit:
        if self.version == 1:
            full_inputs = tf.concat([inputs, comm_features], axis=1)
            final_features, rnn_states_after = self.rnn_cell.call(
                full_inputs, rnn_states
            )
        elif self.version == 2:
            features, rnn_states_after = self.rnn_cell.call(inputs, rnn_states)
            final_features = tf.concat([features, comm_features], axis=1)
        else:
            raise NotImplementedError(self.version)

        # combine observed and communicated information:
        output_features = self.final_layer(final_features)
        out_signal = output_features[:, : self.signal_size]

        # pack for output:
        states_after = rnn_states_after, out_signal
        return output_features, tuple(nest.flatten(states_after))

    def get_visible_ids(self, visible_other_agents):
        result = []
        for ag_id, direction, dist in visible_other_agents:
            result.extend([ag_id, dist, _util.directions_to_i[direction]])
        if not result:
            result = [-1, -1, -1]
        return result

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.comm_rnn.build((input_shape[0], None, self.signal_size))  # + 5))
        if self.version == 1:
            self.rnn_cell.build(
                input_shape[:-1] + (self._comm_out_size + input_shape[-1],)
            )
            self.final_layer.build(input_shape[:-1] + (self.rnn_cell.output_size,))
        elif self.version == 2:
            self.rnn_cell.build(input_shape)
            self.final_layer.build(
                input_shape[:-1]
                + (self._comm_out_size + self.rnn_cell.output_size,)
            )
        self.built = True

    def _get_additional_layers(self):
        return [self.comm_rnn, self.final_layer]

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.signal_size

    @property
    def output_size(self):
        return self.signal_size + self._outputs_size
