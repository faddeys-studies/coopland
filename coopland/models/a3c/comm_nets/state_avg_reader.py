import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseCommCell
from . import _util


class CommCellStateAvgReader(BaseCommCell):
    def __init__(self, rnn_cell):
        super(CommCellStateAvgReader, self).__init__(rnn_cell, 0)

    def call(self, inputs, rnn_states=None, **kwargs):
        assert rnn_states is not None
        inputs, present_indices = inputs
        orig_states = rnn_states
        rnn_states = nest.flatten(rnn_states)

        readed_states = rnn_states[-1]
        readed_states, mask = _util.gather_present(readed_states, present_indices)
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
