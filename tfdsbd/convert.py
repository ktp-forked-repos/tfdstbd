from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import math
import os
import random
import sys
import tensorflow as tf
from functools import partial
from tfucops import expand_split_words, transform_normalize_unicode
from tfdsbd.vocab import Vocabulary
from tfdsbd.model import Model


def tokenize_dataset(raw_paragraphs):
    filter_by_len = partial(filter, len)
    iterator_to_list = partial(map, list)

    paragraphs = raw_paragraphs.split('\n\n')  # 0D -> 1D (text -> paragraphs)
    paragraphs = map(lambda p: p.split('\n'), paragraphs)  # 1D -> 2D (paragraphs -> sentences)
    paragraphs = map(partial(map, lambda s: s.strip('\r\n')), paragraphs)  # strip sentences
    paragraphs = map(filter_by_len, paragraphs)  # filter out 0-len sentences
    paragraphs = iterator_to_list(paragraphs)  # sentence iterator -> sentence list
    paragraphs = filter_by_len(paragraphs)  # filter out 0-len paragraphs
    paragraphs = list(paragraphs)  # paragraph iterator -> paragraph list

    input_ph = tf.placeholder(tf.string, shape=(None))
    tokenize_op = expand_split_words(transform_normalize_unicode(input_ph, 'NFC'))
    with tf.Session() as sess:
        result = [sess.run(tokenize_op, feed_dict={input_ph: sentences}) for sentences in paragraphs]

    result = map(partial(map, filter_by_len), result)  # filter out 0-len tokens
    result = map(partial(map, partial(map, lambda t: t.decode('utf-8'))), result)  # decode binary tokens from UTF-8
    result = map(iterator_to_list, result)  # tokens iterator -> tokens list
    result = iterator_to_list(result)  # sentence iterator -> sentence list
    result = list(result)  # paragraph iterator -> paragraph list

    return result


def make_dataset(tokenized_paragraphs, doc_size, num_repeats):
    def glue(max_spaces, max_tabs, max_newlines):
        glue = [' '] * random.randint(1, max_spaces) + \
               ['\t'] * random.randint(0, max_tabs) + \
               ['\n'] * random.randint(0, max_newlines)

        random.shuffle(glue)
        size = 1 + int(random.expovariate(2.))

        return glue[:size]

    not_boundary_glue = partial(glue, 298, 1, 1)
    default_boundary_glue = partial(glue, 280, 10, 10)
    extra_boundary_glue = partial(glue, 200, 25, 125)

    paragraphs = list(tokenized_paragraphs)
    random.shuffle(paragraphs)
    paragraphs = paragraphs * num_repeats

    documents = []
    labels = []
    while len(paragraphs) > 0:
        sample_size = random.randint(1, doc_size)
        sample, paragraphs = paragraphs[:sample_size], paragraphs[sample_size:]
        sample = list(itertools.chain.from_iterable(sample))  # 3-D list of tokens to 2-D (unpack paragraphs)

        X = []
        y = []
        for sentence in sample:
            sentence = [not_boundary_glue() if token.isspace() else [token] for token in sentence]
            sentence = list(itertools.chain.from_iterable(sentence))

            X_glue = extra_boundary_glue() if sentence[-1][-1].isalnum() else default_boundary_glue()
            y_glue = [0] * (len(X_glue) - 1) + [1]

            X.extend(sentence + X_glue)
            y.extend([0] * len(sentence) + y_glue)
        assert len(X) == len(y), 'items count should be equal labels count'

        document = u''.join(X[:-1])
        document = document.encode('utf-8')  # required due to TF 1.6.0rc1 bug in Python2
        documents.append(document)
        labels.append(y[:-1])

    return list(zip(documents, labels))


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
            for document, labels in rec_data:
                tfrecord_writer.write(create_example(document, labels))
            tf.logging.info('Saved {} examples in {}'.format(len(rec_data), file_name))


def main(argv):
    del argv
    tf.logging.info('Loading source dataset from {}'.format(FLAGS.src_file.name))
    source_content = FLAGS.src_file.read().decode('utf-8')
    tf.logging.info('Source dataset loaded')

    tf.logging.info('Tokenizing source dataset')
    tokenized_samples = tokenize_dataset(source_content)
    tf.logging.info('Source dataset tokenized ({} paragraphs)'.format(len(tokenized_samples)))

    samples_count = len(tokenized_samples)
    valid_count = int(math.floor(samples_count * FLAGS.valid_size))
    test_count = int(math.floor(samples_count * FLAGS.test_size))
    train_count = samples_count - test_count - valid_count

    base_name, _ = os.path.splitext(os.path.basename(FLAGS.src_file.name))

    tf.logging.info('Processing training dataset ({} paragraphs)'.format(train_count))
    train_smaples = tokenized_samples[:train_count]
    train_dataset = make_dataset(train_smaples, FLAGS.doc_size, FLAGS.num_repeats)
    write_dataset(FLAGS.dest_path, 'train', base_name, FLAGS.rec_size, train_dataset)
    tf.logging.info('Training dataset processed')

    tf.logging.info('Processing training vocabulary with min freq = 10')
    words = itertools.chain.from_iterable(train_smaples)  # list of paragraphs to list of sentences
    words = itertools.chain.from_iterable(words)  # list of sentences to list of tokens
    words = list(words)

    vocab = Vocabulary()
    vocab.fit(words)
    vocab.trim(10)
    vocab_filename = os.path.join(FLAGS.dest_path, 'vocabulary')
    vocab.save(vocab_filename + '.pkl')

    tf.logging.info('Vocabulary (as binary) saved to {}'.format(vocab_filename + '.pkl'))

    vocab.fit([Model.UNK_TOKEN, Model.PAD_TOKEN])
    vocab.save(vocab_filename + '.tsv', False)
    tf.logging.info('Vocabulary (as text) saved to {}'.format(vocab_filename + '.tsv'))

    tf.logging.info('Processing validation dataset ({} paragraphs)'.format(valid_count))
    valid_smaples = tokenized_samples[train_count: train_count + valid_count]
    valid_dataset = make_dataset(valid_smaples, FLAGS.doc_size, FLAGS.num_repeats)
    write_dataset(FLAGS.dest_path, 'valid', base_name, FLAGS.rec_size, valid_dataset)
    tf.logging.info('Validation dataset processed')

    tf.logging.info('Processing test dataset ({} paragraphs)'.format(test_count))
    test_smaples = tokenized_samples[train_count + valid_count:]
    test_dataset = make_dataset(test_smaples, FLAGS.doc_size, FLAGS.num_repeats)
    write_dataset(FLAGS.dest_path, 'test', base_name, FLAGS.rec_size, test_dataset)
    tf.logging.info('Test dataset processed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create training/validation/test data from text files with paragraph markup')
    parser.add_argument(
        '-src_file',
        type=argparse.FileType('rb'),
        help=u'Text file with source dataset. Paragraphs should be divided by double \\n, sentences by single one')
    parser.add_argument(
        '-doc_size',
        type=int,
        default=500,
        help='Maximum paragraphs count per document')
    parser.add_argument(
        '-rec_size',
        type=int,
        default=5000,
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
    parser.add_argument(
        '-dest_path',
        type=str,
        help='Directory where to store TFRecord files')

    FLAGS, unparsed = parser.parse_known_args()
    assert FLAGS.valid_size + FLAGS.test_size <= 1
    assert not os.path.exists(FLAGS.dest_path) or os.path.isdir(FLAGS.dest_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
