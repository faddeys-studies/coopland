import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseCommCell
from . import _util


@tf.custom_gradient
def scale_gradient(x, factor):
    def grad(grad_x):
        return factor * grad_x

    return tf.identity(x), grad


class CommCellInnerRNNv3(BaseCommCell):
    def __init__(self, rnn_cell, units, use_gru, use_bidir):
        super(CommCellInnerRNNv3, self).__init__(rnn_cell, 0)
        cell_type = tf.keras.layers.GRUCell if use_gru else tf.keras.layers.LSTMCell
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([cell_type(n) for n in units])
        )
        self._comm_out_size = self.comm_rnn.cell.output_size
        if use_bidir:
            self.comm_rnn = tf.keras.layers.Bidirectional(self.comm_rnn, "sum")

    def call(self, inputs, rnn_states=None, **kwargs):
        assert rnn_states is not None
        inputs, present_indices = inputs
        orig_states = rnn_states
        rnn_states = nest.flatten(rnn_states)

        # unpack info about who and from where is we are communicating
        vis_dists = tf.cast(present_indices[:, 1::3], tf.float32)
        vis_dirs = present_indices[:, 2::3]
        present_indices = present_indices[:, 0::3]
        # n_agents, max_others = tf_utils.get_shape_static_or_dynamic(present_indices)
        vis_dirs_onehot = tf.cast(tf.one_hot(vis_dirs, 4), tf.float32)

        readed_states = rnn_states[-1]
        readed_states, mask = _util.gather_present(readed_states, present_indices)
        readed_states = tf.concat(
            [readed_states, vis_dirs_onehot, tf.expand_dims(vis_dists, 2)], axis=2
        )
        # readed_states = scale_gradient(readed_states, 0.1)
        features = self.comm_rnn.call(readed_states, mask)

        full_input = tf.concat([inputs, features], axis=1)

        features, *rnn_states_after = self.rnn_cell.call(full_input, rnn_states)
        rnn_states_after = nest.pack_sequence_as(
            orig_states, nest.flatten(rnn_states_after)
        )
        return features, rnn_states_after

    def get_visible_ids(self, visible_other_agents):
        result = []
        for ag_id, direction, dist in visible_other_agents:
            result.extend([ag_id, dist, _util.directions_to_i[direction]])
        if not result:
            result = [-1, -1, -1]
        return result

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        read_state_size = nest.flatten(self.rnn_cell.state_size)[-1]
        self.comm_rnn.build((input_shape[0], None, read_state_size + 5))
        self.rnn_cell.build(
            input_shape[:-1] + (self._comm_out_size + input_shape[-1],)
        )
        self.built = True

    def _get_additional_layers(self):
        return [self.comm_rnn]

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size
