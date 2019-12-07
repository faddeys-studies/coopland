import tensorflow as tf
from coopland.models.a3c import config_lib
from .base import BaseCommCell
from . import (
    comm_rnn2,
    comm_rnn1,
    fully_connected,
    comm_net,
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
    elif hparams.comm.type == "comm_rnn1_2":
        return comm_rnn1.CommRNN1(
            cell,
            hparams.comm.units,
            use_gru=hparams.comm.use_gru,
            use_bidir=hparams.comm.use_bidir,
            can_see_others=hparams.comm.can_see_others,
            version=2,
        )
    elif hparams.comm.type == "commrnn2":
        return comm_rnn2.CommRNN2(
            cell,
            hparams.comm.units,
            use_gru=hparams.comm.use_gru,
            use_bidir=hparams.comm.use_bidir,
            can_see_others=hparams.comm.can_see_others,
        )
    raise ValueError(hparams.comm.type)
