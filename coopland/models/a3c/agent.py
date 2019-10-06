import tensorflow as tf
import numpy as np
import dataclasses
from coopland.maze_lib import Direction
from coopland.game_lib import Observation


class TensorflowAgent:
    def __init__(
        self, name, input_tensor, output_tensor, init_fn, input_encoder, output_decoder
    ):
        self.name = name
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.init_fn = init_fn
        self.input_encoder = input_encoder
        self.output_decoder = output_decoder
        self.session = tf.compat.v1.Session()

    def __call__(
        self, agent_id, visibility, corners, visible_other_agents, visible_exit
    ):
        input_data, metadata = self.input_encoder(
            agent_id, visibility, corners, visible_other_agents, visible_exit
        )
        output_data = self.session.run(
            self.output_tensor, {self.input_tensor: [input_data]}
        )
        move = self.output_decoder(output_data, metadata)
        return move

    def init_before_game(self):
        if self.init_fn is not None:
            self.init_fn(self.session)

    def close(self):
        self.session.close()


class AgentModel:

    INPUT_DATA_SIZE = 4 + 8 + 5  # visibility + corners + exit

    def __init__(self):
        self.i_to_direction = Direction.list_clockwise()
        self.directions_to_i = {d: i for i, d in enumerate(self.i_to_direction)}

    def build_layers(self, name=None):
        if name:
            name_prefix = name + "_"
        else:
            name_prefix = ""
        actor = tf.keras.Sequential(
            [
                tf.keras.layers.RNN(
                    [
                        tf.keras.layers.LSTMCell(30),
                        tf.keras.layers.LSTMCell(30),
                        tf.keras.layers.LSTMCell(30),
                    ],
                    return_sequences=True,
                    stateful=True,
                ),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4)),
            ],
            name=name_prefix + "Actor"
        )
        critic = tf.keras.Sequential(
            [
                tf.keras.layers.RNN(
                    [
                        tf.keras.layers.LSTMCell(30),
                        tf.keras.layers.LSTMCell(30),
                        tf.keras.layers.LSTMCell(30),
                    ],
                    return_sequences=True,
                    stateful=True,
                ),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
            ],
            name=name_prefix + "Critic"
        )
        input_ph = tf.compat.v1.placeholder(tf.float32, [1, None, self.INPUT_DATA_SIZE])
        actor(input_ph)
        critic(input_ph)
        return AgentInstance(actor, critic, input_ph)

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

    def decode_nn_output(self, outputs, metadata):
        probs = outputs[0][0, 0]
        value = outputs[1][0, 0]
        input_vector = metadata["input_vector"]
        observation = metadata["full_observation"]
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

    def create_agent_fn(self, agent_instance):
        def init_fn(sess):
            with sess.as_default():
                agent_instance.critic.reset_states()
                agent_instance.actor.reset_states()

        actor_logits, actor_probabilities, critic_tensor = agent_instance()

        agent_fn = TensorflowAgent(
            name="RNN",
            input_tensor=agent_instance.default_input,
            output_tensor=(actor_probabilities, critic_tensor),
            init_fn=init_fn,
            input_encoder=self.encode_observation,
            output_decoder=self.decode_nn_output,
        )
        # .reset_states() creates placeholder and assign nodes when called first time
        # call this now to make sure that these nodes are created in the same graph
        agent_fn.init_before_game()
        return agent_fn


@dataclasses.dataclass
class AgentInstance:
    actor: "tf.keras.Model"
    critic: "tf.keras.Model"
    default_input: "tf.Tensor"
    _default_agent_probabilities: "tf.Tensor" = None
    _default_critic_value: "tf.Tensor" = None

    def __call__(self, input_tensor=None):
        if input_tensor is None:
            actor_logits = self.actor.outputs[0]
            if self._default_critic_value is None:
                self._default_critic_value = self.critic.outputs[0][..., 0]
            critic_value = self._default_critic_value
            if self._default_agent_probabilities is None:
                self._default_agent_probabilities = tf.nn.softmax(actor_logits, axis=-1)
            actor_probabilities = self._default_agent_probabilities
        else:
            actor_logits = self.actor(input_tensor)
            critic_value = self.critic(input_tensor)[..., 0]
            actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)
        return actor_logits, actor_probabilities, critic_value

    def get_variables(self):
        return self.critic.trainable_variables + self.actor.trainable_variables


@dataclasses.dataclass
class Move:
    direction: Direction
    direction_idx: int
    input_vector: np.ndarray
    probabilities: np.ndarray
    critic_value: np.ndarray
    observation: Observation

    def __repr__(self):
        return f"<{self.direction} V={self.critic_value:3f} A={self.probabilities}>"
