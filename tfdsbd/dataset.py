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
from tfunicode import expand_split_words


def parse_dataset(raw_content):
    result_paragraphs = []
    for raw_paragraph in raw_content.strip().split('\n\n'):
        raw_paragraph = raw_paragraph.strip()
        if not len(raw_paragraph):
            continue

        result_sentences = []
        for raw_sentence in raw_paragraph.strip().split('\n'):
            raw_sentence = raw_sentence.strip()
            if not len(raw_sentence):
                continue

            result_sentences.append(raw_sentence)
        result_paragraphs.append(result_sentences)
    np.random.shuffle(result_paragraphs)

    return result_paragraphs


def _make_glue_values(max_spaces, max_tabs, max_newlines, reserve):
    glue_values = [' '] * np.random.randint(1, (max_spaces * 0.95 + 1) * reserve) + \
                  [u'\u00A0'] * np.random.randint(1, (max_spaces * 0.05 + 1) * reserve) + \
                  ['\t'] * np.random.randint(0, (max_tabs + 1) * reserve) + \
                  ['\n'] * np.random.randint(0, int(max_newlines * 0.9 + 1) * reserve) + \
                  ['\r\n'] * np.random.randint(0, int(max_newlines * 0.1 + 1) * reserve)
    np.random.shuffle(glue_values)

    glue_sizes = np.random.exponential(0.5, len(glue_values))

    result = []
    si, vi = 0, 0
    while len(glue_values) > vi and len(glue_sizes) > si:
        size = 1 + int(glue_sizes[si])
        value = glue_values[vi:vi + size]
        si, vi = si + 1, vi + size
        result.append(''.join(value))

    return result


def augment_dataset(source_paragraphs, reserve=10000):
    inner_glue = _make_glue_values(298, 1, 1, reserve * 10)
    default_glue = _make_glue_values(280, 10, 10, reserve)
    extra_glue = _make_glue_values(200, 25, 125, reserve)

    result_paragraphs = []
    for p, raw_paragraph in enumerate(source_paragraphs):
        result_sentences = []
        for raw_sentence in raw_paragraph:
            sentence_glue = extra_glue.pop() if raw_sentence[-1].isalnum() else default_glue.pop()
            if not len(default_glue):
                default_glue = _make_glue_values(280, 10, 10, reserve)
            if not len(extra_glue):
                extra_glue = _make_glue_values(200, 25, 125, reserve)

            spaces = [i for i in range(len(raw_sentence)) if raw_sentence[i].isspace()]
            for space in reversed(spaces):
                words_glue = inner_glue.pop()
                if not len(inner_glue):
                    inner_glue = _make_glue_values(298, 1, 1, reserve * 10)
                if ' ' == inner_glue:
                    continue
                raw_sentence = raw_sentence[:space] + words_glue + raw_sentence[space + 1:]
            result_sentences.append(raw_sentence + sentence_glue)
        result_paragraphs.append(result_sentences)

    return result_paragraphs


def _aling_paragraphs_length(source_paragraphs):
    max_len = max([len(p) for p in source_paragraphs])

    return [p + [''] * (max_len - len(p)) for p in source_paragraphs]


def tokenize_dataset(source_paragraphs, batch_size=1000):
    # Sort by approximate number of tokens for lower memory consumption
    source_paragraphs = sorted(source_paragraphs, key=lambda p: sum([s.count(' ') for s in p]))

    paragraphs_input = tf.placeholder(tf.string, shape=[None, None])
    transform_pipeline = expand_split_words(paragraphs_input)
    transform_pipeline = tf.sparse_tensor_to_dense(transform_pipeline, default_value='')

    result_paragraphs = []
    with tf.Session() as sess:
        while len(source_paragraphs):
            pipeline_todo, source_paragraphs = source_paragraphs[:batch_size], source_paragraphs[batch_size:]
            pipeline_todo = _aling_paragraphs_length(pipeline_todo)

            pipeline_done = sess.run(transform_pipeline, feed_dict={paragraphs_input: pipeline_todo})

            for dp in pipeline_done:
                done_sentences = []
                for ds in dp:
                    ds = [w.decode('utf-8') for w in ds if len(w)]
                    if len(ds):
                        done_sentences.append(ds)
                if len(done_sentences):
                    result_paragraphs.append(done_sentences)

    np.random.shuffle(result_paragraphs)

    return result_paragraphs


def make_dataset(source_paragraphs, doc_size):
    documents = []
    labels = []

    while len(source_paragraphs) > 0:
        sample_words = []
        sample_labels = []

        while len(sample_words) < doc_size and len(source_paragraphs) > 0:
            current_paragraph = source_paragraphs.pop()

            for sentence in current_paragraph:
                if len(sentence) > 1 and sentence[-2] == ' ' and sentence[-1] in ['.', '!', '?']:
                    sentence = sentence[:-2] + sentence[-1:]
                last_meaning = max([i for i, w in enumerate(sentence) if len(w.strip())])

                sample_words.extend(sentence)
                sample_labels.extend(['N'] * (last_meaning + 1))
                sample_labels.extend(['B'] * (len(sentence) - last_meaning - 1))

                assert len(sample_words) == len(sample_labels), 'tokens count should be equal labels count'

        documents.append(''.join(sample_words))
        labels.append(sample_labels)

    dataset = list(zip(documents, labels))
    dataset.sort(key=lambda d: len(d[1]), reverse=True)  # sort from max to min length

    return dataset


def write_dataset(dest_path, set_title, base_name, rec_size, set_data):
    def create_example(document, labels):
        return tf.train.Example(features=tf.train.Features(feature={
            'document': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(document)])),
            'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(l) for l in labels])),
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
    source_content = FLAGS.src_file.read().decode('utf-8')
    base_name, _ = os.path.splitext(os.path.basename(FLAGS.src_file.name))
    paragraphs_data = parse_dataset(source_content)

    tf.logging.info('Splitting train/test/valid datasets')
    samples_count = len(paragraphs_data)
    valid_count = int(math.floor(samples_count * FLAGS.valid_size))
    test_count = int(math.floor(samples_count * FLAGS.test_size))
    train_count = samples_count - test_count - valid_count
    train_paragraphs = paragraphs_data[:train_count] * FLAGS.num_repeats
    valid_paragraphs = paragraphs_data[train_count: train_count + valid_count] * FLAGS.num_repeats
    test_paragraphs = paragraphs_data[train_count + valid_count:] * FLAGS.num_repeats
    del paragraphs_data

    tf.logging.info('Processing test dataset ({} paragraphs)'.format(test_count))
    test_augmented = augment_dataset(test_paragraphs)
    test_tokenized = tokenize_dataset(test_augmented)
    test_dataset = make_dataset(test_tokenized, FLAGS.doc_size)
    write_dataset(FLAGS.dest_path, 'test', base_name, FLAGS.rec_size, test_dataset)
    del test_paragraphs, test_augmented, test_tokenized, test_dataset

    tf.logging.info('Processing valid dataset ({} paragraphs)'.format(valid_count))
    valid_augmented = augment_dataset(valid_paragraphs)
    valid_tokenized = tokenize_dataset(valid_augmented)
    valid_dataset = make_dataset(valid_tokenized, FLAGS.doc_size)
    write_dataset(FLAGS.dest_path, 'valid', base_name, FLAGS.rec_size, valid_dataset)
    del valid_paragraphs, valid_augmented, valid_tokenized, valid_dataset

    tf.logging.info('Processing train dataset ({} paragraphs)'.format(train_count))
    train_augmented = augment_dataset(train_paragraphs)
    train_tokenized = tokenize_dataset(train_augmented)
    train_dataset = make_dataset(train_tokenized, FLAGS.doc_size)
    write_dataset(FLAGS.dest_path, 'train', base_name, FLAGS.rec_size, train_dataset)
    del train_paragraphs, train_augmented, train_tokenized, train_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create training/validation/testing datasets from text files with paragraph markup')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help='Text file with source dataset. Paragraphs should be divided with double \\n, sentences with single one')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory to store TFRecord files')
    parser.add_argument(
        '-doc_size',
        type=int,
        default=500,
        help='Maximum words count per document')
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
        help='How many times repeat each dataset. Useful due sentences shuffling and random glue')

    FLAGS, unparsed = parser.parse_known_args()
    assert not os.path.exists(FLAGS.dest_path) or os.path.isdir(FLAGS.dest_path)
    assert 0 < FLAGS.doc_size
    assert 0 < FLAGS.rec_size
    assert 0 <= FLAGS.valid_size + FLAGS.test_size <= 1
    assert 0 < FLAGS.num_repeats

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
