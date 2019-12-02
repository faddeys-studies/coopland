import tensorflow as tf
from tensorflow.python.util import nest
from coopland import tf_utils
from .base import BaseCommCell
from . import _util


class CommCellSignalAsFeature(BaseCommCell):
    def __init__(self, rnn_cell, signal_size):
        super(CommCellSignalAsFeature, self).__init__(rnn_cell, signal_size)
        self.signal_generator = tf.keras.layers.Dense(
            signal_size, activation=tf.nn.leaky_relu
        )

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, signals = nest.pack_sequence_as(self._state_structure, states)

        assert signals.get_shape().as_list()[1] == self.signal_size

        signals_sets, _ = _util.gather_present(signals, present_indices)
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
            i = _util.directions_to_i[direction]
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
