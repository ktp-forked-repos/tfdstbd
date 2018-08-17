from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import tensorflow as tf
from tfunicode import expand_split_words


def parse_dataset(raw_content, num_repeats=1):
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

    result_paragraphs *= num_repeats
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
                # if len(sentence) > 1 and sentence[-2] == ' ' and sentence[-1] in ['.', '!', '?']:
                #     sentence = sentence[:-2] + sentence[-1:]
                last_meaning = max([i for i, w in enumerate(sentence) if len(w.strip())])

                sample_words.extend(sentence)
                sample_labels.extend(['N'] * (last_meaning + 1))
                sample_labels.extend(['B'] * (len(sentence) - last_meaning - 1))

                assert len(sample_words) == len(sample_labels), 'tokens count should be equal labels count'

        documents.append(''.join(sample_words))
        labels.append(','.join(sample_labels))

    dataset = list(zip(documents, labels))
    dataset.sort(key=lambda d: len(d[1]), reverse=True)  # sort from max to min length

    return dataset


def write_dataset(dest_path, base_name, examples_batch):
    def create_example(document, labels):
        return tf.train.Example(features=tf.train.Features(feature={
            'document': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(document)])),
            'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(labels)])),
        })).SerializeToString()

    if not len(examples_batch):
        return

    try:
        os.makedirs(dest_path)
    except:
        pass

    exist_pattern = '-{}.tfrecords.gz'.format(base_name)
    exist_records = [file.split('-')[0] for file in os.listdir(dest_path) if file.endswith(exist_pattern)]
    next_index = max([int(head) if head.isdigit() else 0 for head in exist_records] + [0]) + 1

    file_name = os.path.join(dest_path, '{}-{}.tfrecords.gz'.format(next_index, base_name))
    rec_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(file_name, options=rec_options) as tfrecord_writer:
        for document, labels in examples_batch:
            tfrecord_writer.write(create_example(document, labels))


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset from text files with paragraph markup')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help='Text file with paragraphs divided by double \\n, sentences by single one')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory to store TFRecord files')
    parser.add_argument(
        '-doc_size',
        type=int,
        default=250,
        help='Maximum words count per document')
    parser.add_argument(
        '-rec_size',
        type=int,
        default=50000,
        help='Maximum documents count per TFRecord file')
    parser.add_argument(
        '-num_repeats',
        type=int,
        default=10,
        help='How many times repeat source data. Useful due paragraphs shuffling and random glue')

    argv, _ = parser.parse_known_args()
    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path)
    assert 0 < argv.doc_size
    assert 0 < argv.rec_size
    assert 0 < argv.num_repeats

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Loading source dataset from {}'.format(argv.src_file.name))
    source_content = argv.src_file.read().decode('utf-8')
    base_name, _ = os.path.splitext(os.path.basename(argv.src_file.name))
    paragraphs_data = parse_dataset(source_content, argv.num_repeats)

    tf.logging.info('Processing dataset ({}K paragraphs)'.format(len(paragraphs_data) // 1000))
    examples_queue = []
    while len(paragraphs_data):
        paragraphs_todo, paragraphs_data = paragraphs_data[:argv.rec_size], paragraphs_data[argv.rec_size:]

        tf.logging.info('Augmenting {}K paragraphs'.format(len(paragraphs_todo) // 1000))
        augmented_todo = augment_dataset(paragraphs_todo)

        tf.logging.info('Tokenizing {}K paragraphs'.format(len(augmented_todo) // 1000))
        tokenized_todo = tokenize_dataset(augmented_todo, batch_size=argv.doc_size)

        tf.logging.info('Converting {}K paragraphs'.format(len(tokenized_todo) // 1000))
        examples_todo = make_dataset(tokenized_todo, doc_size=argv.doc_size)
        examples_queue.extend(examples_todo)

        if len(examples_queue) >= argv.rec_size:
            tf.logging.info('Saving dataset to {}'.format(argv.dest_path))
            examples_todo, examples_queue = examples_queue[:argv.rec_size], examples_queue[argv.rec_size:]
            write_dataset(argv.dest_path, base_name, examples_todo)

    if len(examples_queue):
        tf.logging.info('Saving dataset to {}'.format(argv.dest_path))
        examples_todo, examples_queue = examples_queue[:argv.rec_size], examples_queue[argv.rec_size:]
        write_dataset(argv.dest_path, base_name, examples_todo)
