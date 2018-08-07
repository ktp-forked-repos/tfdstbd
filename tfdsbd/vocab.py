from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from nlpvocab import Vocabulary
from .param import build_hparams
from .input import train_input_fn


def extract_vocab(dest_path, minn, maxn, min_freq):
    wildcard = os.path.join(dest_path, 'train-*.tfrecords.gz')
    dataset = train_input_fn(wildcard, 10, minn, maxn)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    ngrams = next_element[0]['ngrams'].values

    vocab = Vocabulary()
    with tf.Session() as sess:
        while True:
            try:
                result = sess.run(ngrams)
            except tf.errors.OutOfRangeError:
                break
            result = [n.decode('utf-8') for n in result if n != b'' and n != b'<>']
            # only non-alpha, including suffixes and postfixes
            result = [n for n in result if not n.isalpha()]
            vocab.update(result)
            # vocab.trim(2)  # ngrams produce too large vocabulary
    vocab.trim(min_freq)

    return vocab


def main(argv):
    del argv

    params = build_hparams(FLAGS.hyper_params.read())
    assert 0 < params.input_ngram_minn <= params.input_ngram_maxn

    tf.logging.info('Processing training vocabulary with min freq {}'.format(FLAGS.min_freq))
    vocab = extract_vocab(FLAGS.src_path, params.input_ngram_minn, params.input_ngram_maxn, FLAGS.min_freq)

    FLAGS.vocab_file.close()

    vocab.save(FLAGS.vocab_file.name, Vocabulary.FORMAT_BINARY_PICKLE)
    vocab.save(os.path.splitext(FLAGS.vocab_file.name)[0] + '.tsv', Vocabulary.FORMAT_TSV_WITH_HEADERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract ngram vocabulary from training dataset')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path with train TFRecord files')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparams file')
    parser.add_argument(
        'vocab_file',
        type=argparse.FileType('wb'),
        help='Output vocabulary file')
    parser.add_argument(
        '-min_freq',
        type=int,
        default=100,
        help='Minimum ngram frequency to leave it in vocabulary')

    FLAGS, unparsed = parser.parse_known_args()
    assert os.path.exists(FLAGS.src_path)
    assert 0 <= FLAGS.min_freq

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
