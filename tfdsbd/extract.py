from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from tfucops import expand_split_words, transform_normalize_unicode, transform_add_borders, expand_char_ngrams
from .vocab import Vocabulary

def main(argv):
    del argv

    tf.logging.info('Loading source dataset from {}'.format(FLAGS.src_file.name))
    source_content = FLAGS.src_file.read().decode('utf-8')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract token usage statistic from text corpus')
    parser.add_argument(
        'src_path',
        type=argparse.FileType('rb'),
        help=u'Directory with text corpus')
    parser.add_argument(
        'dest_file',
        type=str,
        help='File with extracted statistic')
    parser.add_argument(
        '-min_freq',
        type=int,
        default=10,
        help='Minimum token frequency to leave it in vocabulary')

    FLAGS, unparsed = parser.parse_known_args()
    assert FLAGS.valid_size + FLAGS.test_size <= 1
    assert not os.path.exists(FLAGS.dest_path) or os.path.isdir(FLAGS.dest_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
