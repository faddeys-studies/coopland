import tensorflow as tf
import numpy as np
import dataclasses
import os
from typing import Optional
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation


class AgentModel:
    generates_signals = False
    consumes_signals = False
    signal_size = 0

    def __init__(self):
        self.input_data_size = 4 + 8 + 5  # visibility + corners + exit
        self.i_to_direction = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.i_to_direction)}

    def encode_observation(
        self, agent_id, visibility, corners, visible_other_agents, visible_exit
    ):
        del agent_id, visible_other_agents
        result = []
        result.extend(visibility)
        result.extend(sum(corners, []))
        exit_vec = [0] * 5
        exit_dir, exit_dist = visible_exit
        if exit_dir is not None:
            exit_vec[self.directions_to_i[exit_dir]] = 1
            exit_vec[-1] = exit_dist
        result.extend(exit_vec)
        return np.array(result)

    def _encode_visible_others(self, agent_id, visible_other_agents, max_agents):
        result = [0, 0, 0, 0] * (max_agents - 1)
        for ag_id, direction, dist in visible_other_agents:
            if ag_id >= max_agents:
                continue
            offs = 4 * (ag_id if ag_id < agent_id else ag_id - 1)
            if dist == 0:
                result[offs : offs + 4] = 1, 1, 1, 1
            else:
                i = offs + self.directions_to_i[direction]
                result[i] = 1 / dist
        return result

    def decode_nn_output(
        self, outputs, input_vector, observation, signal, greed_choice_prob=None
    ):
        probs = outputs[0][0, 0]
        value = outputs[1][0, 0]
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
            signal=signal,
        )

    def build_layers(self, name=None):
        if name:
            name_prefix = name + "_"
        else:
            name_prefix = ""

        rnn = self._build_rnn(name_prefix)
        actor_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(4), name=name_prefix + "Actor/Head"
        )
        critic_head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1), name=name_prefix + "Critic/Head"
        )

        rnn.build((None, None, self.input_data_size))
        actor_head.build((None, None, rnn.cell.output_size))
        critic_head.build((None, None, rnn.cell.output_size))

        saver = tf.train.Saver(
            [
                *rnn.trainable_variables,
                *actor_head.trainable_variables,
                *critic_head.trainable_variables,
            ]
        )
        return self._create_agent_instance(rnn, actor_head, critic_head, saver)

    def _create_agent_instance(self, rnn, actor_head, critic_head, saver) -> "AgentInstance":
        raise NotImplementedError

    def _build_rnn(self, name_prefix) -> tf.keras.layers.RNN:
        raise NotImplementedError

    def create_agent_fn(
        self, agent_instance: "AgentInstance", session, greed_choice_prob=None
    ):
        states = {}
        signals = {}
        times = {}

        def agent_fn(*observation):
            agent_id = observation[0]
            t = times.get(agent_id, 0) + 1
            input_vector = self.encode_observation(*observation)
            _run_tensors = (actor_probabilities_t, critic_t), new_states_t

            prev_states = states.get((agent_id, t - 1), [])
            feed = {input_ph: np.reshape(input_vector, (1, 1, -1))}
            feed.update(zip(prev_states_phs, prev_states))
            if self.generates_signals:
                visible_other_agents = observation[3]
                other_agent_ids = [ag_id for ag_id, _, _ in visible_other_agents]
                others_signals = []
                for ag_id in other_agent_ids:
                    others_signal = signals.get((ag_id, t - 1), None)
                    if others_signal is None:
                        others_signal = np.zeros([1, *signals_shape])
                    others_signals.append(others_signal)
                if others_signals:
                    others_signals = np.concatenate(others_signals, axis=0)
                else:
                    others_signals = np.zeros([0, *signals_shape])
                feed[others_signals_ph] = others_signals
                feed[present_indices] = [[np.arange(others_signals.shape[0])]]
                _run_tensors = _run_tensors, out_signal_t

            _out_values = session.run(_run_tensors, feed)
            if self.generates_signals:
                _out_values, out_signal = _out_values
                out_signal = out_signal[:, 0]
                signals[agent_id, t] = out_signal
                signals.pop((agent_id, t - 2), None)
            else:
                out_signal = None
            output_data, new_states = _out_values
            states[agent_id, t] = new_states
            times[agent_id] = t
            states.pop((agent_id, t - 2), None)
            move = self.decode_nn_output(
                output_data, input_vector, observation, out_signal, greed_choice_prob
            )
            return move

        def init_before_game():
            states.clear()
            times.clear()
            signals.clear()

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        input_ph = tf.compat.v1.placeholder(tf.float32, [1, None, self.input_data_size])
        if self.generates_signals:
            others_signals_ph = agent_instance.get_placeholder_for_others_signals()
            signals_shape = others_signals_ph.get_shape().as_list()[1:]
            present_indices = tf.compat.v1.placeholder(tf.int32, [1, 1, None])
        else:
            others_signals_ph = present_indices = None
        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_states_t,
            prev_states_phs,
            out_signal_t,
        ] = agent_instance.call(
            input_ph,
            signals_tensor=others_signals_ph,
            present_indices=present_indices,
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
        signals_tensor=None,
        present_indices: "[N_batch_agents time max_other_agents]" = None,
    ):
        """
        :return: (
            actor_logits: [batch time n_directions],
            actor_probabilities: [batch time n_directions],
            critic_value: [batch time],
            states_after: list of [batch state_size],
            states_before_phs: list of [batch state_size],
            signals: [batch time signal_size],
        )
        """
        raise NotImplementedError

    def get_placeholder_for_others_signals(self) -> Optional[tf.Tensor]:
        pass

    def get_loss_for_signal(
        self, actual_signal, model_signal, advantage
    ) -> Optional[tf.Tensor]:
        pass

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
    signal: Optional[np.ndarray] = None
    debug_text: str = ""

    def __repr__(self):
        return f"<{self.direction} V={self.critic_value:3f} A={self.probabilities}>"


def call_rnn(rnn: tf.keras.layers.RNN, input_tensor, input_mask=None):
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
