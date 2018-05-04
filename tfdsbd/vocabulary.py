from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import open
from collections import Counter
from operator import itemgetter
from six.moves import cPickle
from tfucops import expand_split_words
from .input import train_input_fn
from .transform import extract_ngrams
import argparse
import os
import re
import sys
import tensorflow as tf


class Vocabulary:
    def __init__(self):
        self._cnt = Counter()

    def fit(self, items):
        assert isinstance(items, list), 'items should be a list'
        self._cnt.update(items)

    def trim(self, min_freq):
        for word in list(self._cnt.keys()):
            if self._cnt[word] < min_freq:
                del self._cnt[word]

    def items(self):
        # Due to different behaviour for items with same counts in Python 2 and 3 we should resort result ourselves
        result = self._cnt.most_common()
        if not len(result):
            return []
        result.sort(key=itemgetter(0))
        result.sort(key=itemgetter(1), reverse=True)
        result, _ = zip(*result)

        return list(result)

    def most_common(self, n=None):
        return self._cnt.most_common(n)

    def save(self, filename):
        with open(filename, 'wb') as fout:
            cPickle.dump(self._cnt, fout, protocol=2)

    def export(self, filename, header=True):
        def _safe(word):
            word = u'{}'.format(word)
            if not len(word): return '[]'
            word = re.sub(r'\s', lambda match: '[{}]'.format(ord(match.group(0))), word)
            return word

        with open(filename, 'w', encoding='utf-8') as fout:
            if header:
                line = u'{}\t{}\n'.format(_safe('token'), _safe('frequency'))
                fout.write(line)
            for w in self.items():
                line = u'{}\t{}\n'.format(_safe(w.decode('utf-8')), _safe(self._cnt[w]))
                fout.write(line)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fin:
            cnt = cPickle.load(fin)

        inst = Vocabulary()
        inst._cnt = cnt

        return inst


def extract_vocab(dest_path, minn, maxn, min_freq):
    tf.reset_default_graph()

    wildcard = os.path.join(dest_path, 'train-*.tfrecords.gz')
    dataset = train_input_fn(wildcard, 10)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    docs = next_element[0]['documents']
    tokens = expand_split_words(docs)
    tokens = tf.sparse_tensor_to_dense(tokens, default_value='')
    ngrams = extract_ngrams(tokens, minn, maxn)
    ngrams = tf.reshape(ngrams.values, [-1])

    vocab = Vocabulary()
    with tf.Session() as sess:
        # sess.run(iterator.initializer)
        while True:
            try:
                result = sess.run(ngrams)
            except tf.errors.OutOfRangeError:
                break
            result = [n for n in result if n != b'' and n != b'<>']
            result = [n for n in result if
                      not n.decode('utf-8').isalpha()]  # only non-alpha, including suffixes and postfixes
            vocab.fit(result)
            # vocab.trim(2)  # ngrams produce too large vocabulary
    vocab.trim(min_freq)

    return vocab


def main(argv):
    del argv

    tf.logging.info('Processing training vocabulary with min freq {}'.format(FLAGS.min_freq))
    vocab = extract_vocab(FLAGS.src_path, FLAGS.min_n, FLAGS.max_n, FLAGS.min_freq)
    vocab_filename = os.path.join(FLAGS.src_path, 'vocabulary')
    vocab.save(vocab_filename + '.pkl')
    vocab.fit(['<UNK_{}>'.format(i).encode('utf-8') for i in range(FLAGS.uniq_count)])
    vocab.export(vocab_filename + '.tsv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract vocabulary from training dataset')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path with train TFRecord files')
    parser.add_argument(
        '-min_n',
        type=int,
        default=3,
        help='Minimum ngram size')
    parser.add_argument(
        '-max_n',
        type=int,
        default=4,
        help='Maximum ngram size')
    parser.add_argument(
        '-min_freq',
        type=int,
        default=100,
        help='Minimum ngram frequency to leave it in vocabulary')
    parser.add_argument(
        '-uniq_count',
        type=int,
        default=1000,
        help='Number of vocabulary items to include as <UNK> label in TSV vocabulary')

    FLAGS, unparsed = parser.parse_known_args()
    assert os.path.exists(FLAGS.src_path)
    assert 0 < FLAGS.min_n <= FLAGS.max_n
    assert 0 <= FLAGS.min_freq
    assert 0 < FLAGS.uniq_count

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
