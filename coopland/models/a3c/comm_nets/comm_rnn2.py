import tensorflow as tf

from .base import BaseCommCell
from coopland.models.a3c import util


class CommRNN2(BaseCommCell):
    def __init__(
        self, rnn_cell, units, use_gru, use_bidir, can_see_others, use_comm_init
    ):
        self._can_see_others = can_see_others
        super(CommRNN2, self).__init__(rnn_cell, units[-1])
        cell_type = tf.keras.layers.GRUCell if use_gru else tf.keras.layers.LSTMCell
        self.comm_rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells([cell_type(n) for n in units])
        )
        self._comm_out_size = self.comm_rnn.cell.output_size
        self._env_out_size = self.rnn_cell.output_size
        if use_bidir:
            self.comm_rnn = tf.keras.layers.Bidirectional(self.comm_rnn, "sum")
        if use_comm_init:
            self._self_signal_decoder = tf.keras.layers.Dense(
                util.get_total_size(self.comm_rnn.cell.state_size),
                activation=tf.nn.tanh,
            )
        else:
            self._self_signal_decoder = None

        self._final_layer = tf.keras.layers.Dense(
            self._comm_out_size, activation=tf.nn.leaky_relu
        )

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        rnn_states, signals = states

        if self._can_see_others:
            vis_features = util.build_visible_agents_features(
                comm_directions, comm_distances
            )
            vis_features = util.average_by_mask(
                vis_features, tf.greater_equal(comm_indices, 0)
            )
            inputs = tf.concat([vis_features, inputs], axis=1)
        env_features, rnn_states_after = self.rnn_cell.call(inputs, rnn_states)

        signal_sets, signals_mask = util.gather_present(signals, comm_indices)
        if self._can_see_others:
            signal_sets = util.add_visible_agents_to_each_timestep(
                signal_sets, comm_directions, comm_distances
            )

        n_comm_steps = tf.shape(signal_sets)[1]
        env_features_broadcasted = tf.tile(
            tf.expand_dims(env_features, 1), [1, n_comm_steps, 1]
        )

        full_comm_inputs = tf.concat([signal_sets, env_features_broadcasted], axis=2)

        if self._self_signal_decoder is not None:
            comm_init_raw = tf.concat([env_features, signals], axis=1)
            comm_init = self._self_signal_decoder(comm_init_raw)
            comm_init = util.decode_embedding(comm_init, self.comm_rnn.cell.state_size)
            _constants = ()
        else:
            comm_init = None
            _constants = None
        out_features = self.comm_rnn.call(
            full_comm_inputs,
            signals_mask,
            initial_state=comm_init,
            constants=_constants,
        )
        out_features = self._final_layer(
            tf.concat([env_features, out_features], axis=1)
        )

        return out_features, (rnn_states_after, out_features)

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        input_size = input_shape[-1]
        if self._can_see_others:
            input_size += 5
        self.rnn_cell.build(input_shape[:-1] + (input_size,))
        comm_in_size = self.signal_size
        comm_in_size += self._env_out_size
        if self._self_signal_decoder is not None:
            self._self_signal_decoder.build((input_shape[0], comm_in_size))
        if self._can_see_others:
            comm_in_size += 5
        self.comm_rnn.build((input_shape[0], None, comm_in_size))
        self._final_layer.build(
            (input_shape[0], self._env_out_size + self._comm_out_size)
        )
        self.built = True

    def _get_additional_layers(self):
        if self._self_signal_decoder is not None:
            return [self.comm_rnn, self._final_layer, self._self_signal_decoder]
        else:
            return [self.comm_rnn, self._final_layer]

    @property
    def _state_structure(self):
        return self.rnn_cell.state_size, self.signal_size

    @property
    def output_size(self):
        return self.signal_size
