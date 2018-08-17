from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.training import HParams
from tfseqestimator import RnnType, DenseActivation


def build_hparams(custom_params):
    params = HParams(
        ngram_minn=3,
        ngram_maxn=3,
        ngram_dimension=3,
        ngram_freq=100,
        ngram_oov=1,
        ngram_combiner='sum',
        batch_size=20,
        sequence_dropout=0.,
        context_dropout=0.,
        rnn_type=RnnType.REGULAR_FORWARD_GRU,
        rnn_layers=[1],
        rnn_dropout=0.,
        dense_layers=[0],
        dense_activation=DenseActivation.RELU,
        dense_dropout=0.,
        train_optimizer="Adam",
        learning_rate=0.05,
        loss_reduction=tf.losses.Reduction.SUM
    )
    params.set_hparam('dense_layers', [])

    params.override_from_dict(custom_params)

    return params
