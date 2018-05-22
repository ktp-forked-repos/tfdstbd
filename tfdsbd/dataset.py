from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import math
import numpy as np
import os
import sys
import tensorflow as tf
from functools import partial
from tfucops import expand_split_words, transform_normalize_unicode


def tokenize_dataset(raw_paragraphs):
    filter_by_len = partial(filter, len)
    iterator_to_list = partial(map, list)

    paragraphs = raw_paragraphs.split(b'\n\n')  # 0D -> 1D (text -> paragraphs)
    paragraphs = map(lambda p: p.split(b'\n'), paragraphs)  # 1D -> 2D (paragraphs -> sentences)
    paragraphs = map(partial(map, lambda s: s.strip(b'\r\n')), paragraphs)  # strip sentences
    paragraphs = map(filter_by_len, paragraphs)  # filter out 0-len sentences
    paragraphs = iterator_to_list(paragraphs)  # sentence iterator -> sentence list
    paragraphs = filter_by_len(paragraphs)  # filter out 0-len paragraphs
    paragraphs = list(paragraphs)  # paragraph iterator -> paragraph list

    tf.reset_default_graph()

    input = tf.placeholder(tf.string, shape=(None))
    pipeline = transform_normalize_unicode(input, 'NFC')
    pipeline = expand_split_words(pipeline)
    pipeline = tf.sparse_tensor_to_dense(pipeline, default_value='')
    with tf.Session() as sess:
        result = [sess.run(pipeline, feed_dict={input: sentences}) for sentences in paragraphs]

    result = map(partial(map, filter_by_len), result)  # filter out 0-len tokens
    result = map(iterator_to_list, result)  # tokens iterator -> tokens list
    result = map(filter_by_len, result)  # filter out 0-len sentences
    result = iterator_to_list(result)  # sentence iterator -> sentence list
    result = list(result)  # paragraph iterator -> paragraph list

    return result


def make_dataset(tokenized_paragraphs, doc_size, num_repeats):
    def glue(max_spaces, max_tabs, max_newlines):
        glue = [b' '] * np.random.randint(1, max_spaces) + \
               [b'\t'] * np.random.randint(0, max_tabs) + \
               [b'\n'] * np.random.randint(0, max_newlines)

        np.random.shuffle(glue)
        size = 1 + int(np.random.exponential(2.))

        return glue[:size]

    not_boundary_glue = partial(glue, 298, 1, 1)
    default_boundary_glue = partial(glue, 280, 10, 10)
    extra_boundary_glue = partial(glue, 200, 25, 125)

    paragraphs = list(tokenized_paragraphs)
    paragraphs = paragraphs * num_repeats
    np.random.shuffle(paragraphs)

    documents = []
    labels = []
    while len(paragraphs) > 0:
        sample_size = 1 if doc_size == 1 else np.random.randint(1, doc_size)
        sample, paragraphs = paragraphs[:sample_size], paragraphs[sample_size:]
        sample = list(itertools.chain.from_iterable(sample))  # 3-D list of tokens to 2-D (unpack paragraphs)

        X = []
        y = []
        for sentence in sample:
            if len(sentence) > 1 and sentence[-2] == ' ' and sentence[-1] in ['.', '!', '?']:
                sentence = sentence[:-2] + sentence[-1:]
            sentence = [not_boundary_glue() if token.isspace() else [token] for token in sentence]
            sentence = [token for token in sentence if len(token)]  # filter out empty tokens
            sentence = list(itertools.chain.from_iterable(sentence))

            last_letter = sentence[-1].decode('utf-8')[-1]
            X_glue = extra_boundary_glue() if last_letter.isalnum() else default_boundary_glue()
            y_glue = [1] * len(X_glue)

            X.extend(sentence + X_glue)
            y.extend([0] * len(sentence) + y_glue)
        assert len(X) == len(y), 'items count should be equal labels count'

        documents.append(b''.join(X))
        labels.append(y)

    dataset = list(zip(documents, labels))
    dataset.sort(key=lambda d: len(d[1]), reverse=True)  # sort from max to min len

    return dataset


def write_dataset(dest_path, set_title, base_name, rec_size, set_data):
    def create_example(document, labels):
        return tf.train.Example(features=tf.train.Features(feature={
            'document': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(document)])),
            'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        })).SerializeToString()

    if not len(set_data):
        return

    try:
        os.makedirs(dest_path)
    except:
        pass
    file_mask = os.path.join(dest_path, '{}-{}-{}.tfrecords.gz')

    tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    for i in range(1 + len(set_data) // rec_size):
        rec_data, set_data = set_data[:rec_size], set_data[rec_size:]
        file_name = file_mask.format(set_title, base_name, i)
        with tf.python_io.TFRecordWriter(file_name, options=tfrecord_options) as tfrecord_writer:
            tf.logging.info('Saving {} examples in {}'.format(len(rec_data), file_name))
            for document, labels in rec_data:
                tfrecord_writer.write(create_example(document, labels))


def main(argv):
    del argv

    tf.logging.info('Loading source dataset from {}'.format(FLAGS.src_file.name))
    source_content = FLAGS.src_file.read()
    base_name, _ = os.path.splitext(os.path.basename(FLAGS.src_file.name))

    tf.logging.info('Tokenizing source dataset')
    tokenized_samples = tokenize_dataset(source_content)
    del source_content

    tf.logging.info('Splitting tokenized dataset')
    samples_count = len(tokenized_samples)
    valid_count = int(math.floor(samples_count * FLAGS.valid_size))
    test_count = int(math.floor(samples_count * FLAGS.test_size))
    train_count = samples_count - test_count - valid_count
    train_smaples = tokenized_samples[:train_count]
    valid_smaples = tokenized_samples[train_count: train_count + valid_count]
    test_smaples = tokenized_samples[train_count + valid_count:]
    del tokenized_samples

    tf.logging.info('Processing training dataset ({} paragraphs)'.format(train_count))
    train_dataset = make_dataset(train_smaples, FLAGS.doc_size, FLAGS.num_repeats)
    write_dataset(FLAGS.dest_path, 'train', base_name, FLAGS.rec_size, train_dataset)
    del train_dataset

    tf.logging.info('Processing validation dataset ({} paragraphs)'.format(valid_count))
    valid_dataset = make_dataset(valid_smaples, FLAGS.doc_size, 1)
    write_dataset(FLAGS.dest_path, 'valid', base_name, FLAGS.rec_size, valid_dataset)
    del valid_dataset

    tf.logging.info('Processing test dataset ({} paragraphs)'.format(test_count))
    test_dataset = make_dataset(test_smaples, FLAGS.doc_size, 1)
    write_dataset(FLAGS.dest_path, 'test', base_name, FLAGS.rec_size, test_dataset)
    del test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create training/validation/testing datasets from text files with paragraph markup')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help=u'Text file with source dataset. Paragraphs should be divided by double \\n, sentences by single one')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory where to store TFRecord files')
    parser.add_argument(
        '-doc_size',
        type=int,
        default=10,
        help='Maximum paragraphs count per document')
    parser.add_argument(
        '-rec_size',
        type=int,
        default=10000,
        help='Maximum documents count per TFRecord file')
    parser.add_argument(
        '-valid_size',
        type=float,
        default=0.1,
        help='Proportion of data to include in validation dataset')
    parser.add_argument(
        '-test_size',
        type=float,
        default=0.1,
        help='Proportion of data to include in test dataset')
    parser.add_argument(
        '-num_repeats',
        type=int,
        default=10,
        help='How many times repeat each dataset. Useful due to random sentences glue')

    FLAGS, unparsed = parser.parse_known_args()
    assert not os.path.exists(FLAGS.dest_path) or os.path.isdir(FLAGS.dest_path)
    assert 0 < FLAGS.doc_size
    assert 0 < FLAGS.rec_size
    assert 0 <= FLAGS.valid_size + FLAGS.test_size <= 1
    assert 0 < FLAGS.num_repeats

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
