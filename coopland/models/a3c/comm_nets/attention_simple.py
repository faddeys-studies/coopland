import tensorflow as tf
from tensorflow.python.util import nest
from coopland import tf_utils
from .base import BaseCommCell
from coopland.models.a3c import util


class CommCellAttention(BaseCommCell):
    def __init__(self, rnn_cell, signal_size):
        super(CommCellAttention, self).__init__(rnn_cell, signal_size)
        self._attention = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                signal_size, activation="sigmoid", bias_initializer="ones"
            )
        )

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, present_indices = inputs
        rnn_states, signals = nest.pack_sequence_as(self._state_structure, states)

        # unpack info about who and from where is we are communicating
        vis_dists = tf.cast(present_indices[:, 1::3], tf.float32)
        vis_dirs = present_indices[:, 2::3]
        present_indices = present_indices[:, 0::3]
        vis_dirs_onehot = tf.cast(tf.one_hot(vis_dirs, 4), tf.float32)

        n_agents, max_others = tf_utils.get_shape_static_or_dynamic(present_indices)

        signals_sets, presence_mask = util.gather_present(signals, present_indices)

        attention_inputs = tf.concat(
            [signals_sets, tf.tile(tf.expand_dims(signals, 1), [1, max_others, 1])],
            axis=2,
        )

        presence_mask_float = tf.cast(presence_mask, tf.float32)
        n_others = tf.maximum(tf.reduce_sum(presence_mask_float), 1.0)
        attention = self._attention(attention_inputs)

        comm_features = tf.reduce_sum(attention * signals_sets, axis=1) / n_others

        full_input = tf.concat([inputs, comm_features], axis=1)

        output, rnn_states_after = self.rnn_cell.call(full_input, rnn_states)
        new_own_signal = output[:, :self.signal_size]
        states_after = rnn_states_after, new_own_signal
        full_output = tf.concat([new_own_signal, output], axis=1)
        return full_output, tuple(nest.flatten(states_after))

    def build(self, input_shape):
        n_agents, input_size = input_shape
        n_others = None
        self.rnn_cell.build((n_agents, input_size + self.signal_size))
        self._attention.build((n_agents, n_others, 2 * self.signal_size))
        self.built = True

    def _get_additional_layers(self):
        return [self._attention]
