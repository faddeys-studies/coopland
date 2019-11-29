import tensorflow as tf
import dataclasses
from coopland import config_lib
from coopland.a3c import agent as agent_lib


class AgentModel(agent_lib.AgentModel):
    def __init__(self, hparams: config_lib.ModelHParamsNoCoop):
        super(AgentModel, self).__init__()
        self._hparams = hparams

    def _create_agent_instance(
        self, rnn, actor_head, critic_head, saver
    ) -> "AgentInstance":
        return AgentInstance(rnn, actor_head, critic_head, saver)

    def _build_rnn(self, name_prefix) -> tf.keras.layers.RNN:
        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(units) for units in self._hparams.rnn_units]
        )
        return tf.keras.layers.RNN(
            cell, return_state=True, return_sequences=True, name=name_prefix + "RNN"
        )


@dataclasses.dataclass
class AgentInstance(agent_lib.AgentInstance):
    def call(
        self,
        input_tensor,
        sequence_lengths_tensor=None,
        input_mask=None,
        signals_tensor=None,
        present_indices: "[N_batch_agents time max_other_agents]" = None,
    ):
        if input_mask is None:
            if sequence_lengths_tensor is not None:
                input_mask = agent_lib.build_input_mask(sequence_lengths_tensor)

        features, states_after, states_before_phs = agent_lib.call_rnn(
            self.rnn, input_tensor, input_mask
        )
        actor_logits = self.actor_head(features)
        critic_value = self.critic_head(features)[:, :, 0]
        actor_probabilities = tf.nn.softmax(actor_logits, axis=-1)

        return (
            actor_logits,
            actor_probabilities,
            critic_value,
            states_after,
            states_before_phs,
            None,
        )
