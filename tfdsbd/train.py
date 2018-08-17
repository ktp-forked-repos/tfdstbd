from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from tfseqestimator import SequenceItemsClassifier
from .input import input_feature_columns, train_input_fn, serve_input_fn
from .param import build_hparams


def train_eval_export(ngram_vocab, custom_params, model_path, train_data,
                      eval_data=None, export_path=None, ngram_ckpt=None, eval_first=True):
    # Prepare hyperparameters
    params = build_hparams(custom_params)

    # Prepare sequence estimator
    sequence_feature_columns = input_feature_columns(
        ngram_vocab=ngram_vocab,
        ngram_dimension=params.ngram_dimension,
        ngram_oov=params.ngram_oov,
        ngram_combiner=params.ngram_combiner,
        ngram_ckpt=ngram_ckpt,
    )
    estimator = SequenceItemsClassifier(
        label_vocabulary=['N', 'B'],  # Not a boundary, Boundary
        loss_reduction=params.loss_reduction,
        sequence_columns=sequence_feature_columns,
        sequence_dropout=params.sequence_dropout,
        rnn_type=params.rnn_type,
        rnn_layers=params.rnn_layers,
        rnn_dropout=params.rnn_dropout,
        dense_layers=params.dense_layers,
        dense_activation=params.dense_activation,
        dense_dropout=params.dense_dropout,
        dense_norm=False,
        train_optimizer=params.train_optimizer,
        learning_rate=params.learning_rate,
        model_dir=model_path,
    )

    # Forward splitted words
    estimator = tf.contrib.estimator.forward_features(estimator, 'words')

    # Run training
    train_wildcard = os.path.join(train_data, '*.tfrecords.gz')
    train_steps = 1 if eval_first and not os.path.exists(model_path) else None  # Make evaluation after first step
    estimator.train(input_fn=lambda: train_input_fn(
        wild_card=train_wildcard,
        batch_size=params.batch_size,
        ngram_minn=params.ngram_minn,
        ngram_maxn=params.ngram_maxn
    ), steps=train_steps)

    # Save vocabulary for TensorBoard
    ngram_vocab.update(['<UNK_{}>'.format(i) for i in range(params.ngram_oov)])
    ngram_vocab.save(os.path.join(model_path, 'tensorboard.tsv'), Vocabulary.FORMAT_TSV_WITH_HEADERS)

    # Run evaluation
    metrics = None
    if eval_data is not None:
        eval_wildcard = os.path.join(eval_data, '*.tfrecords.gz')
        metrics = estimator.evaluate(input_fn=lambda: train_input_fn(
            wild_card=eval_wildcard,
            batch_size=params.batch_size,
            ngram_minn=params.ngram_minn,
            ngram_maxn=params.ngram_maxn
        ))

        metrics['f1'] = 0.
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

    if export_path is not None:
        estimator.export_savedmodel(
            export_path, serve_input_fn(
                ngram_minn=params.ngram_minn,
                ngram_maxn=params.ngram_maxn
            ),
            strip_default_attrs=True
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train, evaluate and export tfdsbd model')
    parser.add_argument(
        'train_data',
        type=str,
        help='Directory with TFRecord files for training')
    parser.add_argument(
        'ngram_vocab',
        type=str,
        help='Pickle-encoded ngram vocabulary file')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparameters file')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to store model checkpoints')
    parser.add_argument(
        '-eval_data',
        type=str,
        default=None,
        help='Directory with TFRecord files for evaluation')
    parser.add_argument(
        '-export_path',
        type=str,
        default=None,
        help='Path to store exported model')
    parser.add_argument(
        '-ngram_ckpt',
        type=str,
        default=None,
        help='Checkpoint for ngram embeddings initialization')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.train_data) and os.path.isdir(argv.train_data)
    assert os.path.exists(argv.ngram_vocab) and os.path.isfile(argv.ngram_vocab)
    assert not os.path.exists(argv.model_path) or os.path.isdir(argv.model_path)
    assert argv.eval_data is None or os.path.exists(argv.eval_data) and os.path.isdir(argv.eval_data)
    assert argv.export_path is None or not os.path.exists(argv.export_path) or os.path.isdir(argv.export_path)
    assert argv.ngram_ckpt is None or os.path.exists(argv.ngram_ckpt) and os.path.isdir(argv.ngram_ckpt)

    tf.logging.set_verbosity(tf.logging.INFO)

    # Load vocabulary
    ngram_vocab = Vocabulary.load(argv.ngram_vocab, format=Vocabulary.FORMAT_BINARY_PICKLE)

    # Load hyperparams
    custom_params = json.loads(argv.hyper_params.read())

    metrics = train_eval_export(
        ngram_vocab=ngram_vocab,
        custom_params=custom_params,
        model_path=argv.model_path,
        train_data=argv.train_data,
        eval_data=argv.eval_data,
        export_path=argv.export_path,
        ngram_ckpt=argv.ngram_ckpt,
        eval_first=True,
    )
    if metrics is not None:
        print(metrics)
