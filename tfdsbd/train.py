# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from nlpvocab import Vocabulary
from tf1 import f1_score
from tfseqestimator import SequenceItemsClassifier
from tensorflow.python.estimator.canned import prediction_keys
from .input import input_feature_columns, train_input_fn  # , serve_input_fn
from .param import build_hparams


def main(argv):
    del argv

    # Load vocabulary
    ngram_vocab = Vocabulary.load(FLAGS.ngram_vocab, format=Vocabulary.FORMAT_BINARY_PICKLE)

    # Prepare hyperparameters
    params = build_hparams(FLAGS.hyper_params.read())

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
        model_params={
            'sequence_dropout': params.model_sequence_dropout,
            'context_dropout': params.model_context_dropout,
            'rnn_type': params.model_rnn_type,
            'rnn_layers': params.model_rnn_layers,
            'rnn_dropout': params.model_rnn_dropout,
            'dense_layers': params.model_dense_layers,
            'dense_activation': params.model_dense_activation,
            'dense_dropout': params.model_dense_dropout,
            'train_optimizer': params.train_train_optimizer,
            'learning_rate': params.train_learning_rate,
        },
        sequence_columns=sequence_feature_columns,
        model_dir=FLAGS.model_path,
    )

    # Add F1 metric
    def additional_metrics(labels, predictions):
        print(predictions)
        return {'f1': f1_score(labels, predictions[prediction_keys.PredictionKeys.CLASSES])}

    estimator = tf.contrib.estimator.add_metrics(estimator, additional_metrics)

    # Forward splitted words
    # https://towardsdatascience.com/how-to-extend-a-canned-tensorflow-estimator-to-add-more-evaluation-metrics-and-to-pass-through-ddf66cd3047d
    estimator = tf.contrib.estimator.forward_features(estimator, 'words_out')


    # Run training
    # hook = tf.train.ProfilerHook(save_steps=2, output_dir=FLAGS.model_path, show_memory=True)
    train_wildcard = os.path.join(FLAGS.data_path, 'train*.tfrecords.gz')
    estimator.train(input_fn=lambda: train_input_fn(
        wild_card=train_wildcard,
        batch_size=params.input_batch_size,
        ngram_minn=params.input_ngram_minn,
        ngram_maxn=params.input_ngram_maxn
    ))

    # Save vocabulary for TensorBoard
    ngram_vocab.update(['<UNK_{}>'.format(i) for i in range(params.input_ngram_oov)])
    ngram_vocab.save(os.path.join(FLAGS.model_path, 'vocabulary.tsv'), Vocabulary.FORMAT_TSV_WITHOUT_HEADERS)

    # Run evaluation
    eval_wildcard = os.path.join(FLAGS.data_path, 'valid*.tfrecords.gz')
    metrics = estimator.evaluate(input_fn=lambda: train_input_fn(
        wild_card=eval_wildcard,
        batch_size=params.input_batch_size,
        ngram_minn=params.input_ngram_minn,
        ngram_maxn=params.input_ngram_maxn
    ))
    print(metrics)

    # if len(FLAGS.export_path):
    #     # feature_inputs = {
    #     #     'age': tf.placeholder(dtype=tf.float32, shape=[1, 1], name='age'),
    #     # tf.estimator.export.build_raw_serving_input_receiver_fn(feature_inputs)
    #     estimator.export_savedmodel(FLAGS.export_path, serve_input_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train, evaluate and export SBD model')
    parser.add_argument(
        'data_path',
        type=str,
        help='Directory with TFRecord files')
    parser.add_argument(
        'ngram_vocab',
        type=str,
        help='Pickle-encoded ngram vocabulary file')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparams file')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to store model checkpoints')
    parser.add_argument(
        '-export_path',
        type=str,
        default='',
        help='Path to store exported model')

    FLAGS, unparsed = parser.parse_known_args()
    assert os.path.exists(FLAGS.data_path) and os.path.isdir(FLAGS.data_path)
    assert os.path.exists(FLAGS.ngram_vocab) and os.path.isfile(FLAGS.ngram_vocab)
    assert not os.path.exists(FLAGS.model_path) or os.path.isdir(FLAGS.model_path)
    assert not os.path.exists(FLAGS.export_path) or os.path.isdir(FLAGS.export_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
