from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tensorflow as tf
from nlpvocab import Vocabulary
from .param import build_hparams
from .input import train_input_fn


def extract_vocab(dest_path, minn, maxn, min_freq, batch_size):
    wildcard = os.path.join(dest_path, '*.tfrecords.gz')
    dataset = train_input_fn(wildcard, batch_size, minn, maxn)
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
            # only non-alpha, including suffixes, postfixes and other interesting parts
            result = [n for n in result if not n.isalpha()]
            vocab.update(result)
    vocab.trim(min_freq)

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description='Extract ngram vocabulary from dataset')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path with train TFRecord files')
    parser.add_argument(
        'hyper_params',
        type=argparse.FileType('rb'),
        help='JSON-encoded model hyperparams file')
    parser.add_argument(
        'ngram_vocab',
        type=str,
        help='Output vocabulary file')
    parser.add_argument(
        '-min_freq',
        type=int,
        default=100,
        help='Minimum ngram frequency to leave it in vocabulary')

    argv, unparsed = parser.parse_known_args()
    assert os.path.exists(argv.src_path) and os.path.isdir(argv.src_path)
    assert 0 <= argv.min_freq

    tf.logging.set_verbosity(tf.logging.INFO)

    params = build_hparams(json.loads(argv.hyper_params.read()))
    assert 0 < params.ngram_minn <= params.ngram_maxn

    tf.logging.info('Processing training vocabulary with min freq {}'.format(argv.min_freq))
    vocab = extract_vocab(argv.src_path, params.ngram_minn, params.ngram_maxn, argv.min_freq, params.batch_size)

    vocab.save(argv.ngram_vocab, Vocabulary.FORMAT_BINARY_PICKLE)
    vocab.save(os.path.splitext(argv.ngram_vocab)[0] + '.tsv', Vocabulary.FORMAT_TSV_WITH_HEADERS)
