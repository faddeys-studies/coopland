import tensorflow as tf
import numpy as np
import dataclasses
import os
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation
from coopland.models.a3c import config_lib
from coopland import tf_utils


class AgentModel:
    def __init__(self, hparams: config_lib.AgentModelHParams):
        self.hparams = hparams
        self.input_data_size = 4 + 8 + 5  # visibility + corners + exit
        if self.hparams.use_visible_agents:
            # +4 distances to other agents
            self.input_data_size += 4 * (hparams.max_agents - 1)
        self.i_to_direction = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.i_to_direction)}

    def encode_observation(
        self, agent_id, visibility, corners, visible_other_agents, visible_exit
    ):
        full_observation = (
            agent_id,
            visibility,
            corners,
            visible_other_agents,
            visible_exit,
        )
        result = []
        result.extend(visibility)
        result.extend(sum(corners, []))
        exit_vec = [0] * 5
        exit_dir, exit_dist = visible_exit
        if exit_dir is not None:
            exit_vec[self.directions_to_i[exit_dir]] = 1
            exit_vec[-1] = exit_dist
        result.extend(exit_vec)

        if self.hparams.use_visible_agents:
            visible_agents_part = [0, 0, 0, 0] * (self.hparams.max_agents - 1)
            for ag_id, direction, dist in visible_other_agents:
                if ag_id >= self.hparams.max_agents:
                    continue
                offs = 4 * (ag_id if ag_id < agent_id else ag_id - 1)
                if dist == 0:
                    visible_agents_part[offs : offs + 4] = 1, 1, 1, 1
                else:
                    i = offs + self.directions_to_i[direction]
                    visible_agents_part[i] = 1 / dist
            result.extend(visible_agents_part)

        vector = np.array(result)
        assert vector.shape == (self.input_data_size,)
        return (
            np.expand_dims(vector, 0),
            {"input_vector": vector, "full_observation": full_observation},
        )

    def decode_nn_output(self, outputs, metadata, greed_choice_prob=None):
        probs = outputs[0][0, 0]
        value = outputs[1][0, 0]
        input_vector = metadata["input_vector"]
        observation = metadata["full_observation"]
        if greed_choice_prob is not None and np.random.random() < greed_choice_prob:
            direction_i = np.argmax(probs)
        else:
            direction_i = np.random.choice(range(len(probs)), p=probs)
        direction = self.i_to_direction[direction_i]
        return Move(
            direction=direction,
            direction_idx=direction_i,
            input_vector=input_vector,
            probabilities=probs,
            critic_value=value,
            observation=observation,
        )

    def build_layers(self, name=None):
        if name:
            name_prefix = name + "_"
        else:
            name_prefix = ""

        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units) for units in self.hparams.rnn_units]
        )
        if self.hparams.use_communication:
            cell = RNNCellWithStateCommunication(
                cell, self.hparams.comm_units, (len(self.hparams.rnn_units) - 1, 0)
            )
        rnn = tf.keras.layers.RNN(
            cell, return_state=True, return_sequences=True, name=name_prefix + "RNN"
        )
        actor_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(4), name=name_prefix + "Actor/Head"
        )
        critic_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1), name=name_prefix + "Critic/Head"
        )

        rnn.build((None, None, self.input_data_size))
        actor_head.build((None, None, self.hparams.rnn_units[-1]))
        critic_head.build((None, None, self.hparams.rnn_units[-1]))

        saver = tf.train.Saver(
            [
                *rnn.trainable_variables,
                *actor_head.trainable_variables,
                *critic_head.trainable_variables,
            ]
        )
        return AgentInstance(rnn, actor_head, critic_head, saver)

    def create_agent_fn(
        self, agent_instance: "AgentInstance", session, greed_choice_prob=None
    ):
        states = {}
        times = {}

        def agent_fn(*observation):
            agent_id = observation[0]
            t = times.get(agent_id, 0) + 1
            input_data, metadata = self.encode_observation(*observation)

            prev_states = states.get((agent_id, t - 1), [])
            feed = {input_ph: [input_data]}
            feed.update(zip(prev_states_phs, prev_states))
            if self.hparams.use_communication:
                rnn_cell: RNNCellWithStateCommunication = agent_instance.rnn.cell
                visible_other_agents = observation[3]
                other_agent_states = []
                for ag_id, _, _ in visible_other_agents:
                    other_prev_state = states.get((ag_id, t - 1), [])
                    if other_prev_state:
                        state_to_add = rnn_cell.get_comm_state(other_prev_state)
                    else:
                        state_to_add = np.zeros([1, rnn_cell.comm_state_size])
                    other_agent_states.append(state_to_add)
                if other_agent_states:
                    other_agent_states = np.concatenate(other_agent_states, axis=0)
                else:
                    other_agent_states = np.zeros([1, rnn_cell.comm_state_size])
                feed[others_state_ph] = other_agent_states

            output_data, new_states = session.run(
                ((actor_probabilities_t, critic_t), new_states_t), feed
            )
            states[agent_id, t] = new_states
            times[agent_id] = t
            states.pop((agent_id, t - 2), None)
            move = self.decode_nn_output(output_data, metadata, greed_choice_prob)
            return move

        def init_before_game():
            states.clear()

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        input_ph = tf.compat.v1.placeholder(tf.float32, [1, None, self.input_data_size])
        if self.hparams.use_communication:
            assert isinstance(agent_instance.rnn.cell, RNNCellWithStateCommunication)
            others_state_ph = tf.compat.v1.placeholder(
                tf.float32, [None, agent_instance.rnn.cell.comm_state_size]
            )
            others_indices = tf.reshape(
                tf.range(tf.shape(others_state_ph)[0]), [1, 1, -1]
            )
        else:
            others_state_ph = others_indices = None
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_states_t,
            prev_states_phs,
        ] = agent_instance.call(
            input_ph,
            const_others_state_tensor=others_state_ph,
            others_indices=others_indices,
        )
        del actor_logits_t

        return agent_fn


@dataclasses.dataclass
class AgentInstance:
    rnn: "tf.keras.layers.RNN"
    actor_head: "tf.keras.layers.Layer"
    critic_head: "tf.keras.layers.Layer"
    saver: "tf.train.Saver"

    def call(
        self,
        input_tensor,
        sequence_lengths_tensor=None,
        input_mask=None,
        const_others_state_tensor=None,
        others_indices=None,
    ):
        if input_mask is None:
            if sequence_lengths_tensor is not None:
                input_mask = build_input_mask(sequence_lengths_tensor)

        use_communication = isinstance(self.rnn.cell, RNNCellWithStateCommunication)
        if use_communication:
            self.rnn.cell.const_others_state_tensor = const_others_state_tensor

            # shape of `others_indices` = (n_agents, time, n_others)
            # shape of `input_tensor` = (n_agents, time, state_size)
            # insert pseudo-steps for doing the communication
            # and then remove those steps when no communication or real step is done
            shp = tf_utils.get_shape_static_or_dynamic(others_indices)
            n_agents, n_timesteps, n_others = shp

            def insert_pseudo_steps(tensor, time_axis):
                ax = time_axis
                orig_shape = tf_utils.get_shape_static_or_dynamic(tensor)
                rank = len(orig_shape)
                tensor = tf.transpose(tensor, (*range(ax), *range(ax+1, rank), ax))
                tensor = tf.expand_dims(tensor, axis=-1)
                tensor = tf.pad(
                    tensor,
                    [(0, 0)] * rank + [(n_others - 1, 0)],
                )
                tensor = tf.reshape(
                    tensor, orig_shape[:ax] + orig_shape[ax+1:] + [-1]
                )
                tensor = tf.transpose(tensor, (*range(ax), rank-1, *range(ax, rank-1)))
                return tensor

            input_tensor = insert_pseudo_steps(input_tensor, 1)
            real_step_flags = insert_pseudo_steps(tf.ones([n_timesteps], bool), 0)
            others_indices_inlined = tf.reshape(others_indices, [n_agents, -1])
            if input_mask is not None:
                input_mask = tf.tile(input_mask, [1, n_others, 1])

            has_comm_mask = tf.reduce_any(tf.not_equal(others_indices_inlined, -1), 0)
            keep_mask = tf.logical_or(has_comm_mask, real_step_flags)

            input_tensor = tf.boolean_mask(input_tensor, keep_mask, axis=1)
            real_step_flags = tf.boolean_mask(real_step_flags, keep_mask, axis=0)
            others_indices_inlined = tf.boolean_mask(
                others_indices_inlined, keep_mask, axis=1
            )
            if input_mask is not None:
                input_mask = tf.boolean_mask(input_mask, keep_mask, axis=1)

            real_step_flags = tf.reshape(real_step_flags, [1, -1, 1])
            others_indices_inlined = tf.expand_dims(others_indices_inlined, 2)
            input_tensor = input_tensor, others_indices_inlined, real_step_flags
        else:
            real_step_flags = None
        features, states_after, states_before_phs = _call_rnn(
            self.rnn, input_tensor, input_mask
        )
        if use_communication:
            self.rnn.cell.const_others_state_tensor = None
            features = tf.boolean_mask(features, real_step_flags[0, :, 0], axis=1)
        actor_logits = self.actor_head(features)
        critic_value = self.critic_head(features)[:, :, 0]
        actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)

        return (
            actor_logits,
            actor_probabilities,
            critic_value,
            states_after,
            states_before_phs,
        )

    def get_variables(self):
        layers = self.rnn, self.actor_head, self.critic_head
        return [v for layer in layers for v in layer.trainable_variables]

    def save_variables(self, session, directory, step):
        checkpoint_path = os.path.join(directory, "model.ckpt")
        self.saver.save(
            session, checkpoint_path, global_step=step, write_meta_graph=False
        )

    def load_variables(self, session, directory):
        save_path = tf.train.latest_checkpoint(directory)
        self.saver.restore(session, save_path)
        step = int(save_path.rpartition("-")[2])
        return step


@dataclasses.dataclass
class Move:
    direction: Direction
    direction_idx: int
    input_vector: np.ndarray
    probabilities: np.ndarray
    critic_value: np.ndarray
    observation: Observation
    debug_text: str = ""

    def __repr__(self):
        return f"<{self.direction} V={self.critic_value:3f} A={self.probabilities}>"


def _call_rnn(rnn: tf.keras.layers.RNN, input_tensor, input_mask=None):
    if isinstance(input_tensor, (list, tuple)):
        some_input = input_tensor[0]
    else:
        some_input = input_tensor
    if some_input.get_shape().as_list()[0] is None:
        batch_size = tf.shape(some_input)[0]
        batch_size_value = None
    else:
        batch_size = batch_size_value = some_input.get_shape().as_list()[0]
    state_phs = nest.map_structure(
        lambda size: tf.compat.v1.placeholder_with_default(
            tf.zeros([batch_size, size]), [batch_size_value, size]
        ),
        rnn.cell.state_size,
    )
    state_keras_inputs = nest.map_structure(
        lambda ph: tf.keras.Input(tensor=ph), state_phs
    )
    keras_inputs = nest.map_structure(lambda t: tf.keras.Input(tensor=t), input_tensor)

    out, *states = rnn.call(
        keras_inputs, initial_state=state_keras_inputs, mask=input_mask, constants=()
    )

    state_phs_flat = nest.flatten(state_phs)
    states_flat = nest.flatten(states)

    return out, states_flat, state_phs_flat


def build_input_mask(sequence_lengths_tensor):
    batch_size = tf.shape(sequence_lengths_tensor)[0]
    max_seq_len = tf.reduce_max(sequence_lengths_tensor)
    step_indices = tf.range(max_seq_len)
    step_indices = tf.tile(tf.expand_dims(step_indices, 0), [batch_size, 1])
    mask = tf.less(step_indices, tf.expand_dims(sequence_lengths_tensor, 1))
    mask = tf.expand_dims(mask, axis=2)
    return mask


class RNNCellWithStateCommunication(tf.keras.layers.Layer):
    def __init__(self, cell, comm_net_units, application_path):
        super(RNNCellWithStateCommunication, self).__init__()
        self.cell: tf.keras.layers.Layer = cell

        self._application_path = tuple(application_path)
        self._flat_idx = list(nest.yield_flat_paths(self.cell.state_size)).index(
            self._application_path
        )
        self.comm_state_size = _getpath(self.cell.state_size, application_path)
        self._zero_comm_state_for_mask = tf.zeros([self.comm_state_size])
        assert isinstance(self.comm_state_size, int)
        self._comm_net_units = [*comm_net_units, self.comm_state_size]
        self._comm_net = [
            tf.keras.layers.Dense(
                units, activation=tf.nn.leaky_relu, kernel_initializer="zeros"
            )
            for units in self._comm_net_units
        ]

        self.const_others_state_tensor = None

    def get_comm_state(self, states):
        return states[self._flat_idx]

    def build(self, input_shape):
        self.cell.build(input_shape)
        comm_input_units = 2 * self.comm_state_size
        for layer, units in zip(self._comm_net, self._comm_net_units):
            layer.build(comm_input_units)
            comm_input_units = units
        self.built = True

    @property
    def state_size(self):
        return self.cell.state_size

    def call(self, inputs, states=None, constants=None, **kwargs):
        assert states is not None
        inputs, visible_states_indices, real_step = inputs
        visible_states_indices = tf.squeeze(visible_states_indices, axis=1)
        real_step = tf.reshape(real_step, ())
        others_state_tensor = self.const_others_state_tensor
        if others_state_tensor is None:
            others_state_tensor = _getpath(states, self._application_path)
        init_comm_states = _getpath(states, self._application_path)

        others_state_tensor_with_pad = tf.pad(others_state_tensor, [[1, 0], [0, 0]])

        comm_states = tf.concat(
            [
                init_comm_states,
                tf.gather(others_state_tensor_with_pad, visible_states_indices + 1),
            ],
            axis=-1,
        )
        for layer in self._comm_net:
            comm_states = layer(comm_states)

        final_comm_states = tf.where(
            tf.greater_equal(visible_states_indices, 0),
            init_comm_states + comm_states,
            init_comm_states,
        )

        paths = list(nest.yield_flat_paths(states))
        states_flat = nest.flatten(states)
        states_flat[paths.index(self._application_path)] = final_comm_states
        states_after_comm = nest.pack_sequence_as(states, states_flat)

        outputs, new_states = self.cell.call(
            inputs, states_after_comm, constants, **kwargs
        )
        new_states = nest.map_structure(
            lambda new_st, st: tf.where(real_step, new_st, st),
            new_states, states_after_comm
        )
        return outputs, new_states

    @property
    def trainable_weights(self):
        if self.trainable:
            return [
                *self.cell.trainable_weights,
                *(v for layer in self._comm_net for v in layer.trainable_weights),
            ]
        else:
            return []


def _getpath(obj, path):
    for item in path:
        obj = obj[item]
    return obj
