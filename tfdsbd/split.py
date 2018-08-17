from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import numpy as np
import tensorflow as tf


def split_paragraphs(source_paragraphs, test_size):
    source_paragraphs = [p.strip() for p in source_paragraphs]
    source_paragraphs = [p for p in source_paragraphs if len(p)]
    np.random.shuffle(source_paragraphs)

    paragraphs_count = len(source_paragraphs)
    test_count = int(math.floor(paragraphs_count * test_size))
    test_paragraphs, train_paragraphs = source_paragraphs[:test_count], source_paragraphs[test_count:]

    return train_paragraphs, test_paragraphs


def main():
    parser = argparse.ArgumentParser(
        description='Split source paragraphs into train and test parts')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help='Path to source paragraphs file')
    parser.add_argument(
        'train_file',
        type=argparse.FileType('wb'),
        help='Path to train paragraphs file')
    parser.add_argument(
        'test_file',
        type=argparse.FileType('wb'),
        help='Path to test paragraphs file')
    parser.add_argument(
        '-test_size',
        type=float,
        default=0.05,
        help='Proportion of data to include in test dataset')

    argv, _ = parser.parse_known_args()
    assert 0 < argv.test_size < 1

    tf.logging.set_verbosity(tf.logging.INFO)

    source_paragraphs = argv.src_file.read().decode('utf-8').split('\n\n')
    train_paragraphs, test_paragraphs = split_paragraphs(source_paragraphs, argv.test_size)

    train_raw = '\n\n'.join(train_paragraphs) + '\n\n'
    argv.train_file.write(train_raw.encode('utf-8'))

    test_raw = '\n\n'.join(test_paragraphs) + '\n\n'
    argv.test_file.write(test_raw.encode('utf-8'))