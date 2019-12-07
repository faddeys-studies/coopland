import tensorflow as tf
from coopland.models.a3c import config_lib
from .base import BaseCommCell
from . import (
    comm_rnn2,
    comm_rnn1,
    fully_connected,
    comm_net,
    attention,
    attention_simple,
)


def create(hparams: config_lib.AgentModelHParams, cell: tf.keras.layers.Layer):
    net = _create(hparams, cell)
    print("Created CommCell with params:", hparams, "->", net)
    return net


def _create(hparams: config_lib.AgentModelHParams, cell: tf.keras.layers.Layer):
    if hparams.comm.type == "fully_connected":
        return fully_connected.CommCellFullyConnected(
            cell,
            max_agents=hparams.max_agents,
            signal_size=hparams.comm.units[0],
            can_see_others=hparams.comm.can_see_others,
        )
    if hparams.comm.type == "comm_net":
        return comm_net.CommNetCell(
            cell,
            hparams.comm.units[0] if hparams.comm.units else None,
            hparams.comm.can_see_others,
        )
    elif hparams.comm.type == "comm_rnn1_1":
        return comm_rnn1.CommRNN1(
            cell,
            hparams.comm.units,
            use_gru=hparams.comm.use_gru,
            use_bidir=hparams.comm.use_bidir,
            can_see_others=hparams.comm.can_see_others,
            version=1,
        )
    elif hparams.comm_type == "inner_rnn_v1":
        return comm_rnn2.CommCellInnerRNN(
            cell,
            hparams.comm_units,
            hparams.comm_dropout_rate,
            1,
            use_gru=hparams.use_gru,
            use_bidir=hparams.use_bidir,
            see_others=hparams.see_others,
        )
    elif hparams.comm_type == "signal_averager":
        return comm_net.CommNetCell(
            cell, hparams.comm_units[-1], can_see_others=hparams.see_others
        )
    elif hparams.comm_type == "inner_rnn_v3":
        return comm_rnn1.CommRNN1(
            cell, hparams.comm_units, hparams.use_gru, hparams.use_bidir
        )
    elif hparams.comm_type == "attention":
        return attention.CommCellAttention(cell, hparams.comm_units[0])
    elif hparams.comm_type == "attention_simple":
        return attention_simple.CommCellAttention(cell, hparams.comm_units[0])
    raise ValueError(hparams.comm_type)
