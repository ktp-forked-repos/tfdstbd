from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from nlpvocab import Vocabulary
from tfseqestimator import SequenceItemsClassifier
from .input import input_feature_columns, train_input_fn, serve_input_fn
from .param import build_hparams


def train_eval_export(ngram_vocab, raw_params, data_path, model_path, export_path, train_repeat=1, eval_first=True):
    # Prepare hyperparameters
    params = build_hparams(raw_params)

    # Prepare sequence estimator
    sequence_feature_columns = input_feature_columns(
        ngram_vocab=ngram_vocab,
        ngram_dimension=params.input_ngram_dimension,
        ngram_oov=params.input_ngram_oov,
        ngram_combiner=params.input_ngram_combiner
    )
    estimator = SequenceItemsClassifier(
        label_vocabulary=['N', 'B'],  # Not a boundary, Boundary
        loss_reduction=params.train_loss_reduction,
        sequence_columns=sequence_feature_columns,
        sequence_dropout=params.model_sequence_dropout,
        rnn_type=params.model_rnn_type,
        rnn_layers=params.model_rnn_layers,
        rnn_dropout=params.model_rnn_dropout,
        dense_layers=params.model_dense_layers,
        dense_activation=params.model_dense_activation,
        dense_dropout=params.model_dense_dropout,
        dense_norm=False,
        train_optimizer=params.train_train_optimizer,
        learning_rate=params.train_learning_rate,
        model_dir=model_path,
    )

    # Forward splitted words
    estimator = tf.contrib.estimator.forward_features(estimator, 'words')

    # Run training
    train_wildcard = os.path.join(data_path, 'train*.tfrecords.gz')
    for _ in range(train_repeat):
        train_steps = 1 if eval_first and not os.path.exists(model_path) else None  # Make evaluation after first step
        estimator.train(input_fn=lambda: train_input_fn(
            wild_card=train_wildcard,
            batch_size=params.input_batch_size,
            ngram_minn=params.input_ngram_minn,
            ngram_maxn=params.input_ngram_maxn
        ), steps=train_steps)

    # Save vocabulary for TensorBoard
    ngram_vocab.update(['<UNK_{}>'.format(i) for i in range(params.input_ngram_oov)])
    ngram_vocab.save(os.path.join(model_path, 'vocabulary.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)

    # Run evaluation
    eval_wildcard = os.path.join(data_path, 'valid*.tfrecords.gz')
    metrics = estimator.evaluate(input_fn=lambda: train_input_fn(
        wild_card=eval_wildcard,
        batch_size=params.input_batch_size,
        ngram_minn=params.input_ngram_minn,
        ngram_maxn=params.input_ngram_maxn
    ))
    metrics['f1'] = 0.
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

    if len(export_path):
        estimator.export_savedmodel(
            export_path, serve_input_fn(
                ngram_minn=params.input_ngram_minn,
                ngram_maxn=params.input_ngram_maxn
            ),
            strip_default_attrs=True
        )

    return metrics