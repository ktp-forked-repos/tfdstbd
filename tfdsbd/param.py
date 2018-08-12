from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.training import HParams
from tfseqestimator import RnnType, DenseActivation


def build_hparams(custom_params):
    params = HParams(
        input_ngram_minn=3,
        input_ngram_maxn=3,
        input_ngram_dimension=3,
        input_ngram_oov=1,
        input_ngram_combiner='sum',
        input_batch_size=20,
        model_sequence_dropout=0.,
        model_context_dropout=0.,
        model_rnn_type=RnnType.REGULAR_FORWARD_GRU,
        model_rnn_layers=[1],
        model_rnn_dropout=0.,
        model_dense_layers=[0],
        model_dense_activation=DenseActivation.RELU,
        model_dense_dropout=0.,
        train_train_optimizer="Adam",
        train_learning_rate=0.05,
        train_loss_reduction=tf.losses.Reduction.SUM
    )
    params.set_hparam('model_dense_layers', [])

    params.override_from_dict(custom_params)

    return params
