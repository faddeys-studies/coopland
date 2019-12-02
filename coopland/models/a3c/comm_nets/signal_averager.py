import tensorflow as tf
from tensorflow.python.util import nest
from .base import BaseCommCell
from . import _util


class CommCellSignalAverager(BaseCommCell):
    def __init__(self, rnn_cell, signal_size):
        super(CommCellSignalAverager, self).__init__(rnn_cell, signal_size)

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, signals = nest.pack_sequence_as(self._state_structure, states)

        signals, mask = _util.gather_present(signals, present_indices)
        mask_float = tf.expand_dims(tf.cast(mask, tf.float32), 2)
        avg_signals = tf.reduce_sum(mask_float * signals, axis=1) / (
            1e-5 + tf.reduce_sum(mask_float, axis=1)
        )

        full_input = tf.concat([inputs, avg_signals], axis=1)

        features, *rnn_states_after = self.rnn_cell.call(full_input, rnn_states)
        out_signal = features[:, : self.signal_size]
        states_after = rnn_states_after, out_signal
        full_out_features = tf.concat([out_signal, features], axis=1)
        return full_out_features, tuple(nest.flatten(states_after))

    def get_visible_ids(self, visible_other_agents):
        return [ag_id for ag_id, direction, dist in visible_other_agents]

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.rnn_cell.build(input_shape[:-1] + (input_shape[-1] + self.signal_size,))
        self.built = True

    def _get_additional_layers(self):
        return []

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.signal_size