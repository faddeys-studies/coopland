import tensorflow as tf
from coopland.models.a3c import config_lib
from .base import BaseCommCell
from . import (
    inner_rnn,
    inner_rnn_v3,
    signal_as_feature,
    signal_averager,
    state_avg_reader,
    attention,
    attention_simple,
)


def create(hparams: config_lib.AgentModelHParams, cell: tf.keras.layers.Layer):
    if hparams.comm_type in (None, "signal_as_feature"):
        return signal_as_feature.CommCellSignalAsFeature(cell, hparams.comm_units[0])
    elif hparams.comm_type == "state_avg_reader":
        return state_avg_reader.CommCellStateAvgReader(cell)
    elif hparams.comm_type in ("inner_rnn", "inner_rnn_v2"):
        return inner_rnn.CommCellInnerRNN(
            cell,
            hparams.comm_units,
            hparams.comm_dropout_rate,
            2,
            use_gru=hparams.use_gru,
            use_bidir=hparams.use_bidir,
        )
    elif hparams.comm_type == "inner_rnn_v1":
        return inner_rnn.CommCellInnerRNN(
            cell,
            hparams.comm_units,
            hparams.comm_dropout_rate,
            1,
            use_gru=hparams.use_gru,
            use_bidir=hparams.use_bidir,
        )
    elif hparams.comm_type == "signal_averager":
        return signal_averager.CommCellSignalAverager(cell, hparams.comm_units[-1])
    elif hparams.comm_type == "inner_rnn_v3":
        return inner_rnn_v3.CommCellInnerRNNv3(
            cell, hparams.comm_units, hparams.use_gru, hparams.use_bidir
        )
    elif hparams.comm_type == "attention":
        return attention.CommCellAttention(cell, hparams.comm_units[0])
    elif hparams.comm_type == "attention_simple":
        return attention_simple.CommCellAttention(cell, hparams.comm_units[0])
    raise ValueError(hparams.comm_type)
