# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from .input import train_input_fn
from .param import build_hparams


def main(argv):
    del argv

    # Prepare hyperparameters
    params = build_hparams(FLAGS.hyper_params.read())

    # Run training
    # hook = tf.train.ProfilerHook(save_steps=2, output_dir=FLAGS.model_path, show_memory=True)
    train_wildcard = os.path.join(FLAGS.data_path, 'train*.tfrecords.gz')
    dataset = train_input_fn(
        wild_card=train_wildcard,
        batch_size=1,
        ngram_minn=params.input_ngram_minn,
        ngram_maxn=params.input_ngram_maxn
    )
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    with tf.Session() as sess:
        while True:
            words_value, labels_value = sess.run([features['words_out'], labels])
            if len(words_value[0]) != len(labels_value[0]):
                # print([[w.decode('utf-8') for w in words_value[0]], labels_value[0]])
                for i in range(min(len(words_value[0]), len(labels_value[0]))):
                    print([words_value[0][i].decode('utf-8'), labels_value[0][i].decode('utf-8')])
                break


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
