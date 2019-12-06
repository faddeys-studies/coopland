import tensorflow as tf
from .base import BaseCommCell
from coopland.models.a3c import util


class CommCellFullyConnected(BaseCommCell):
    def __init__(self, rnn_cell, max_agents, signal_size, can_see_others):
        super(CommCellFullyConnected, self).__init__(rnn_cell, signal_size)
        if self.signal_size > 0:
            self.signal_generator = tf.keras.layers.Dense(
                signal_size, activation=tf.nn.leaky_relu
            )
        else:
            self.signal_generator = None
        signal_size = self.signal_size
        if can_see_others:
            signal_size += 5
        self.signal_matrix_size = max_agents * signal_size
        self.max_agents = max_agents
        self.can_see_others = can_see_others

    def _call(self, inputs, states, comm_indices, comm_directions, comm_distances):
        n_agents, max_others = util.get_shape_static_or_dynamic(comm_indices)
        if self.signal_size > 0:
            rnn_states, signals = states
        else:
            rnn_states = states
            signals = tf.zeros([n_agents, 0])

        signal_sets, present_mask = util.gather_present(signals, comm_indices)
        if self.can_see_others:
            signal_sets = util.add_visible_agents_to_each_timestep(
                signal_sets, comm_directions, comm_distances
            )

        signal_sets = signal_sets[:, : self.max_agents, :]
        signal_sets = tf.pad(
            signal_sets,
            [
                (0, 0),
                (0, self.max_agents - max_others),
                (0, 0),
            ],
        )
        signals_flat = tf.reshape(signal_sets, [n_agents, self.signal_matrix_size])

        full_inputs = tf.concat([inputs, signals_flat], axis=1)
        features, new_rnn_states = self.rnn_cell.call(full_inputs, rnn_states)
        if self.signal_size > 0:
            new_own_signal = self.signal_generator.call(features)
            out_states = new_rnn_states, new_own_signal
        else:
            out_states = new_rnn_states
        return features, out_states

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self.rnn_cell.build(
            input_shape[:-1] + (input_shape[-1] + self.signal_matrix_size,)
        )
        if self.signal_generator is not None:
            self.signal_generator.build(input_shape[:-1] + (self.rnn_cell.output_size,))
        self.built = True

    def _get_additional_layers(self):
        return [self.signal_generator] if self.signal_generator else []

    @property
    def _state_structure(self):
        if self.signal_size > 0:
            return self.rnn_cell.state_size, self.signal_size
        else:
            return self.rnn_cell.state_size

    def get_signal(self, states):
        if self.signal_size > 0:
            return super(CommCellFullyConnected, self).get_signal(states)
        else:
            return None
