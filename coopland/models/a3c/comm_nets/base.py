import tensorflow as tf
from tensorflow.python.util import nest


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
        return [ag_id for ag_id, direction, dist in visible_other_agents]
