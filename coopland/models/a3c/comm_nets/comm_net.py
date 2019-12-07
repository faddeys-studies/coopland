import tensorflow as tf
from tensorflow.python.util import nest
from .base import BaseCommCell
from coopland.models.a3c import util


class CommNetCell(BaseCommCell):
    def __init__(self, rnn_cell, encoder_size, can_see_others):
        super(CommNetCell, self).__init__(rnn_cell, None)
        self.can_see_others = can_see_others
        self._encoder_size = encoder_size
        self._encoder = (
            tf.keras.layers.Dense(encoder_size, activation="tanh")
            if encoder_size is not None
            else None
        )
        self._comm_mats = None

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        if self.can_see_others:
            vis_features = util.build_visible_agents_features(
                comm_directions, comm_distances
            )
            vis_features = _average_by_mask(
                vis_features, tf.greater_equal(comm_indices, 0)
            )
            inputs = tf.concat([vis_features, inputs], axis=1)

        data = self._encoder(inputs) if self._encoder is not None else inputs
        comm = tf.zeros_like(data)

        new_states = []
        for cell, state, comm_mat in zip(self.rnn_cell.cells, states, self._comm_mats):
            data, new_state = cell.call(data + comm_mat(comm), state)
            new_states.append(new_state)
            comm = _gather_and_average(data, comm_indices)

        return data, new_states

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        in_size = input_shape[-1]
        if self.can_see_others:
            in_size += 5
        if self._encoder_size is not None:
            self._encoder.build(input_shape[:-1] + (in_size,))
            in_size = self._encoder.units
        self.rnn_cell.build(input_shape[:-1] + (in_size,))

        rnn_out_units = [cell.output_size for cell in self.rnn_cell.cells]
        self._comm_mats = []
        for units in [in_size] + rnn_out_units[:-1]:
            comm_mat = tf.keras.layers.Dense(units, use_bias=False)
            comm_mat.build(input_shape[:-1] + (units,))
            self._comm_mats.append(comm_mat)
        self.built = True

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size

    def get_signal(self, states):
        return None

    def _get_additional_layers(self):
        assert self.built
        layers = [*self._comm_mats]
        if self._encoder is not None:
            layers.append(self._encoder)
        return layers


def _gather_and_average(
    vectors: "n_agents vector_size", indices: "n_agents max_others"
):
    vector_seqs, mask = util.gather_present(vectors, indices)
    return _average_by_mask(vector_seqs, mask)


def _average_by_mask(vector_seqs: "n_agents time vector_size", mask: "n_agents time"):
    mask_float = tf.expand_dims(tf.cast(mask, tf.float32), 2)
    return tf.reduce_sum(mask_float * vector_seqs, axis=1) / (
        1e-5 + tf.reduce_sum(mask_float, axis=1)
    )
