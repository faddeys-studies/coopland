import tensorflow as tf
from tensorflow.python.util import nest


class BaseCommCell(tf.keras.layers.Layer):
    signal_size = 0

    def __init__(self, wrapped_cell, signal_size):
        super(BaseCommCell, self).__init__()
        self.rnn_cell: tf.keras.layers.StackedRNNCells = wrapped_cell
        self.signal_size = signal_size

    def call(self, inputs, states=None, **kwargs):
        assert states is not None
        inputs, comm_indices, comm_dirs, comm_dists = inputs
        states = nest.pack_sequence_as(self._state_structure, states)

        outputs, new_states = self._call(
            inputs, states, comm_indices, comm_dirs, comm_dists
        )

        return outputs, tuple(nest.flatten(new_states))

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        raise NotImplementedError

    @property
    def output_size(self):
        return self.rnn_cell.output_size

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

    def get_signal(self, states):
        # not that this shall be re-implemented if self._state_structure is overrode
        rnn_states, signal = nest.pack_sequence_as(self._state_structure, states)
        return signal
