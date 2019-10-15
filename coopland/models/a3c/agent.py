import tensorflow as tf
import numpy as np
import dataclasses
import os
from typing import List
from tensorflow.python.util import nest
from coopland.maze_lib import Direction
from coopland.game_lib import Observation


class AgentModel:

    INPUT_DATA_SIZE = 4 + 8 + 5  # visibility + corners + exit

    def __init__(self):
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

        vector = np.array(result)
        assert vector.shape == (self.INPUT_DATA_SIZE,)
        assert vector.dtype == np.int
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

        input_ph = tf.compat.v1.placeholder(tf.float32, [1, None, self.INPUT_DATA_SIZE])

        actor, actor_state_phs = _build_model_with_states(
            tf.keras.layers.RNN(
                StackedLSTMCells(
                    [
                        tf.keras.layers.LSTMCell(100),
                        tf.keras.layers.LSTMCell(100),
                        tf.keras.layers.LSTMCell(100),
                        tf.keras.layers.LSTMCell(100),
                    ]
                ),
                return_state=True,
                return_sequences=True,
                name=name_prefix + "Actor/RNN",
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(4), name=name_prefix + "Actor/Head"
            ),
            input_ph,
        )
        critic, critic_state_phs = _build_model_with_states(
            tf.keras.layers.RNN(
                StackedLSTMCells(
                    [
                        tf.keras.layers.LSTMCell(100),
                        tf.keras.layers.LSTMCell(100),
                        tf.keras.layers.LSTMCell(100),
                        tf.keras.layers.LSTMCell(100),
                    ]
                ),
                return_state=True,
                return_sequences=True,
                name=name_prefix + "Critic/RNN",
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1), name=name_prefix + "Critic/Head"
            ),
            input_ph,
        )
        saver = tf.train.Saver(
            [*actor.trainable_variables, *critic.trainable_variables]
        )
        return AgentInstance(
            actor, critic, input_ph, actor_state_phs, critic_state_phs, saver
        )

    def create_agent_fn(
        self, agent_instance: "AgentInstance", session, greed_choice_prob=None
    ):
        actor_states = []
        critic_states = []

        def agent_fn(*observation):
            input_data, metadata = self.encode_observation(*observation)

            feed = {agent_instance.input_ph: [input_data]}
            feed.update(zip(agent_instance.actor_state_phs, actor_states))
            feed.update(zip(agent_instance.critic_state_phs, critic_states))

            output_data, (new_actor_states, new_critic_states) = session.run(
                (
                    (actor_probabilities_t, critic_t),
                    (new_actor_states_t, new_critic_states_t),
                ),
                feed,
            )
            actor_states[:] = new_actor_states
            critic_states[:] = new_critic_states
            move = self.decode_nn_output(output_data, metadata, greed_choice_prob)
            return move

        def init_before_game():
            actor_states.clear()
            critic_states.clear()

        agent_fn.init_before_game = init_before_game
        agent_fn.name = "RNN"

        [
            actor_logits_t,
            actor_probabilities_t,
            critic_t,
            new_actor_states_t,
            new_critic_states_t,
        ] = agent_instance()
        del actor_logits_t

        return agent_fn


@dataclasses.dataclass
class AgentInstance:
    actor: "tf.keras.Model"
    critic: "tf.keras.Model"
    input_ph: "tf.Tensor"
    actor_state_phs: "List[tf.Tensor]"
    critic_state_phs: "List[tf.Tensor]"
    saver: "tf.train.Saver"
    _default_agent_probabilities: "tf.Tensor" = None
    _default_critic_value: "tf.Tensor" = None

    def __call__(self, input_tensor=None, actor_states=None, critic_states=None):
        if input_tensor is None:
            assert actor_states is None
            assert critic_states is None
            actor_logits = self.actor.outputs[0]
            if self._default_critic_value is None:
                self._default_critic_value = self.critic.outputs[0][..., 0]
            critic_value = self._default_critic_value
            if self._default_agent_probabilities is None:
                self._default_agent_probabilities = tf.nn.softmax(actor_logits, axis=-1)
            actor_probabilities = self._default_agent_probabilities

            actor_after_states = self.actor.outputs[1:]
            critic_after_states = self.critic.outputs[1:]
        else:
            if actor_states is None:
                actor_states = self.actor_state_phs
            if critic_states is None:
                critic_states = self.critic_state_phs
            actor_logits, *actor_after_states = self.actor(
                [input_tensor, *actor_states]
            )
            critic_value, *critic_after_states = self.critic(
                [input_tensor, *critic_states]
            )
            critic_value = critic_value[..., 0]
            actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)
        return (
            actor_logits,
            actor_probabilities,
            critic_value,
            actor_after_states,
            critic_after_states,
        )

    def get_variables(self):
        return self.critic.trainable_variables + self.actor.trainable_variables

    def save_variables(self, session, directory, step):
        checkpoint_path = os.path.join(directory, "model.ckpt")
        self.saver.save(
            session, checkpoint_path, global_step=step, write_meta_graph=False
        )
        with open(os.path.join(directory, "step.txt"), "w") as f:
            f.write(str(step))

    def load_variables(self, session, directory):
        save_path = tf.train.latest_checkpoint(directory)
        self.saver.restore(session, save_path)
        with open(os.path.join(directory, "step.txt"), "r") as f:
            return int(f.read().strip())


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


def _build_model_with_states(rnn, post_layer, input_tensor):
    state_phs = nest.map_structure(
        lambda size: tf.compat.v1.placeholder_with_default(
            tf.zeros([1, size]), [1, size]
        ),
        rnn.cell.state_size,
    )
    state_keras_inputs = nest.map_structure(
        lambda ph: tf.keras.Input(tensor=ph), state_phs
    )
    input_layer = tf.keras.Input(tensor=input_tensor)

    features, *states = rnn(input_layer, initial_state=state_keras_inputs)
    out = post_layer(features)

    state_phs_flat = nest.flatten(state_phs)
    states_flat = nest.flatten(states)
    # states_flat = [st[-1] for st in states_flat]

    model = tf.keras.Model(
        inputs=[input_tensor, *state_phs_flat], outputs=[out, *states_flat]
    )
    return model, state_phs_flat


class StackedLSTMCells(tf.keras.layers.StackedRNNCells):

    _flatten_state_size = True

    @property
    def state_size(self):
        state_size = super(StackedLSTMCells, self).state_size
        if self._flatten_state_size:
            state_size = nest.flatten(state_size)
        return state_size

    def call(self, inputs, states, constants=None, **kwargs):
        flatten_state_size = self._flatten_state_size
        self._flatten_state_size = False
        try:
            return super(StackedLSTMCells, self).call(
                inputs, states, constants=None, **kwargs
            )
        finally:
            self._flatten_state_size = flatten_state_size
