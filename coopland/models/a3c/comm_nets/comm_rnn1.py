import tensorflow as tf
from tensorflow.python.util import nest

from .base import BaseCommCell
from coopland.models.a3c import util


class CommRNN1(BaseCommCell):
    def __init__(
        self,
        rnn_cell,
        units,
        use_gru,
        use_bidir,
        can_see_others,
        version,
        use_comm_init,
    ):
        super(CommRNN1, self).__init__(rnn_cell, 0)
        assert version in (1, 2)
        cell_type = tf.keras.layers.GRUCell if use_gru else tf.keras.layers.LSTMCell
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([cell_type(n) for n in units])
        )
        self.signal_size = self.rnn_cell.output_size if self.version == 2 else units[-1]
        self._can_see_others = can_see_others
        self.version = version
        self._comm_out_size = self.comm_rnn.cell.output_size
        if use_bidir:
            self.comm_rnn = tf.keras.layers.Bidirectional(self.comm_rnn, "sum")

        if use_comm_init:
            self._self_signal_decoder = tf.keras.layers.Dense(
                util.get_total_size(self.comm_rnn.cell.state_size),
                activation=tf.nn.tanh,
            )
        else:
            self._self_signal_decoder = None

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        if self.version == 1:
            signals = states[-1][0]
        else:
            states, signals = states

        signal_seqs, mask = util.gather_present(signals, comm_indices)
        if self._can_see_others:
            signal_seqs = util.add_visible_agents_to_each_timestep(
                signal_seqs, comm_directions, comm_distances
            )
        if self._self_signal_decoder is not None:
            comm_init = self._self_signal_decoder(signals)
            comm_init = util.decode_embedding(comm_init, self.comm_rnn.cell.state_size)
            _constants = ()
        else:
            comm_init = None
            _constants = None
        comm_result = self.comm_rnn.call(
            signal_seqs, mask, initial_state=comm_init, constants=_constants
        )

        if self._can_see_others:
            vis_features = util.build_visible_agents_features(
                comm_directions, comm_distances
            )
            vis_features = util.average_by_mask(
                vis_features, tf.greater_equal(comm_indices, 0)
            )
            inputs = tf.concat([vis_features, inputs], axis=1)
        full_input = tf.concat([inputs, comm_result], axis=1)

        features, *states_after = self.rnn_cell.call(full_input, states)
        if self.version == 2:
            states_after = states_after, features
        return features, states_after

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        comm_in_size = self.get_signal(self._state_structure)
        if self._self_signal_decoder is not None:
            self._self_signal_decoder.build((input_shape[0], comm_in_size))
        if self._can_see_others:
            comm_in_size += 5
        self.comm_rnn.build((input_shape[0], None, comm_in_size))
        in_size = input_shape[-1]
        if self._can_see_others:
            in_size += 5
        self.rnn_cell.build(input_shape[:-1] + (self._comm_out_size + in_size,))
        self.built = True

    def _get_additional_layers(self):
        if self._self_signal_decoder is not None:
            return [self.comm_rnn, self._self_signal_decoder]
        else:
            return [self.comm_rnn]

    @property
    def _state_structure(self):
        if self.version == 1:
            return self.rnn_cell.state_size
        else:
            return self.rnn_cell.state_size, self.signal_size

    def get_signal(self, states):
        states = nest.pack_sequence_as(self._state_structure, nest.flatten(states))
        if self.version == 1:
            return states[-1][0]
        else:
            states, signals = states
            return signals
