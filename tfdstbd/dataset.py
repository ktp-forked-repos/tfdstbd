from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import conllu
import numpy as np
import os
import six
import tensorflow as tf
from itertools import cycle
from tfunicode import expand_split_words


def parse_conllu(file_name):
    with open(file_name, 'rb') as f:
        content = '\n' + f.read().decode('utf-8')

    content = content.replace('\n# newdoc\n', '\n# newdoc = Yes\n')
    content = content.replace('\n# newpar\n', '\n# newpar = Yes\n')
    has_groups = False

    result = []
    paragraph = []
    for block in conllu.parse(content):
        meta = ' '.join(six.iterkeys(block.metadata))

        start_group = 'newdoc' in meta or 'newpar' in meta
        has_groups = has_groups or start_group

        if len(paragraph) and (not has_groups or start_group):
            result.append(paragraph)
            paragraph = []

        sentence, skip = [], []
        for i, token in enumerate(block):
            if isinstance(token['id'], tuple):
                skip.extend(token['id'])

            if token['id'] in skip:
                continue

            sentence.append(token['form'])
            if len(block) - 1 == i:
                continue

            if 'misc' not in token or token['misc'] is None or 'SpaceAfter' not in token['misc']:
                sentence.append(' ')
                continue

            if 'No' == token['misc']['SpaceAfter']:
                continue

            space = token['misc']['SpaceAfter']
            assert space.startswith('\\')
            sentence.append(space.encode('utf-8').decode('unicode_escape'))

        if len(sentence):
            paragraph.append(sentence)

    if len(paragraph):
        result.append(paragraph)

    return result


def random_glue(space=0, tab=0, newline=0, empty=0, reserve=0):
    max_spaces0 = int((space + 1) * 0.95 * reserve)
    num_spaces0 = np.random.randint(0, max_spaces0) if max_spaces0 > 0 else 0

    max_spaces1 = int((space + 1) * 0.05 * reserve)
    num_spaces1 = np.random.randint(0, max_spaces1) if max_spaces1 > 0 else 0

    max_tabs = int((tab + 1) * reserve)
    num_tabs = np.random.randint(0, max_tabs) if max_tabs > 0 else 0

    max_newlines0 = int((newline + 1) * 0.9 * reserve)
    num_newlines0 = np.random.randint(0, max_newlines0) if max_newlines0 > 0 else 0

    max_newlines1 = int((newline + 1) * 0.1 * reserve)
    num_newlines1 = np.random.randint(0, max_newlines1) if max_newlines1 > 0 else 0

    max_empties = int((empty + 1) * reserve)
    num_empties = np.random.randint(0, max_empties) if max_empties > 0 else 0

    glue_values = [' '] * num_spaces0 + \
                  [u'\u00A0'] * num_spaces1 + \
                  ['\t'] * num_tabs + \
                  ['\n'] * num_newlines0 + \
                  ['\r\n'] * num_newlines1 + \
                  [''] * num_empties
    np.random.shuffle(glue_values)
    glue_sizes = np.random.exponential(0.5, len(glue_values))

    result = []
    si, vi = 0, 0
    while len(glue_values) > vi and len(glue_sizes) > si:
        size = 1 + int(glue_sizes[si])
        value = glue_values[vi:vi + size]
        si, vi = si + 1, vi + size
        result.append(value)

    return result


def augment_dataset(source_paragraphs):
    reserve = sum([len(p) for p in source_paragraphs])
    inner_glue = cycle(random_glue(space=297, tab=1, newline=1, empty=1, reserve=reserve * 10))
    outer_glue = cycle(random_glue(space=279, tab=10, newline=10, empty=1, reserve=reserve))
    extra_glue = cycle(random_glue(space=200, tab=25, newline=125, empty=0, reserve=max(10, reserve // 10)))

    result_paragraphs = []
    for p, raw_paragraph in enumerate(source_paragraphs):
        result_sentences = []
        for raw_sentence in raw_paragraph:
            spaces = [i for i in range(len(raw_sentence)) if raw_sentence[i].isspace()]
            for space in reversed(spaces):
                words_glue = next(inner_glue)
                if ' ' == words_glue:
                    continue
                if space > 0 and raw_sentence[space - 1].isalnum() and '' == words_glue[0]:
                    continue
                raw_sentence = raw_sentence[:space] + words_glue + raw_sentence[space + 1:]
            sentence_glue = next(extra_glue) if raw_sentence[-1].isalnum() else next(outer_glue)

            result_sentences.append(raw_sentence + sentence_glue)
        result_paragraphs.append(result_sentences)

    del inner_glue, outer_glue, extra_glue

    return result_paragraphs


def tokenize_sentence(target, side):
    assert ''.join(target) == ''.join(side)
    if not len(target):
        return []

    side_len = [len(w) for w in side]
    side_acc = [sum(side_len[:i]) for i in range(len(side_len))]
    target_len = [len(w) for w in target]
    target_acc = [sum(target_len[:i]) for i in range(len(target_len))]

    same_split = set(side_acc).intersection(target_acc)
    # Break label if same break in target and side at the same time
    target_labels = ['B' if sum(target_len[:i]) in same_split else 'N' for i in range(len(target_len))]
    assert len(target) == len(target_labels)

    return target_labels


def tokenize_dataset(source_paragraphs, batch_size=1000):
    # Sort by approximate number of tokens for lower memory consumption
    source_paragraphs = sorted(source_paragraphs, key=lambda p: sum([s.count(' ') for s in p]))

    transform_input = tf.placeholder(tf.string, shape=[None, None])
    transform_pipeline = expand_split_words(transform_input, extended=True)
    transform_pipeline = tf.sparse_tensor_to_dense(transform_pipeline, default_value='')

    result_paragraphs = []
    with tf.Session() as sess:
        bs = 1
        while len(source_paragraphs):
            pipeline_todo, source_paragraphs = source_paragraphs[:bs], source_paragraphs[bs:]

            # Smaller batch size for longer sentences
            bs = min(bs * 2, batch_size)

            # Join tokens into sentences
            pipeline_input = [[''.join(s) for s in p] for p in pipeline_todo]

            # Align paragraphs length
            max_len = max([len(p) for p in pipeline_todo])
            pipeline_input = [p + [''] * (max_len - len(p)) for p in pipeline_input]

            pipeline_done = sess.run(transform_pipeline, feed_dict={transform_input: pipeline_input})
            assert len(pipeline_done) == len(pipeline_todo)

            for dp, sp in zip(pipeline_done, pipeline_todo):
                dp = [ds for ds in dp if len(b''.join(ds)) > 0]

                assert len(dp) == len(sp), (dp, sp)
                done_paragraph = []

                for ds, ss in zip(dp, sp):
                    ds = [w.decode('utf-8') for w in ds if len(w)]

                    labels = ','.join(tokenize_sentence(ds, ss))
                    done_paragraph.append((ds, labels))

                if len(done_paragraph):
                    result_paragraphs.append(done_paragraph)

    return result_paragraphs


def make_documents(source_paragraphs, doc_size):
    documents = []
    tokens = []
    labels = []

    while len(source_paragraphs) > 0:
        sample_words = []
        sample_tokens = []
        sample_labels = []

        while len(sample_words) < doc_size and len(source_paragraphs) > 0:
            current_paragraph = source_paragraphs.pop()

            for words, breaks in current_paragraph:
                last_meaning = max([i for i, w in enumerate(words) if len(w.strip())])
                sample_words.extend(words)
                sample_tokens.append(breaks)
                sample_labels.extend(['N'] * (last_meaning + 1))
                sample_labels.extend(['B'] * (len(words) - last_meaning - 1))
                assert len(sample_words) == len(sample_labels)

        documents.append(''.join(sample_words))
        tokens.append(','.join(sample_tokens))
        labels.append(','.join(sample_labels))

    assert len(tokens) == len(labels)
    dataset = list(zip(documents, tokens, labels))
    dataset.sort(key=lambda d: len(d[2]), reverse=True)  # sort from max to min length

    return dataset


def write_dataset(dest_path, base_name, examples_batch):
    def create_example(document, tokens, sentences):
        return tf.train.Example(features=tf.train.Features(feature={
            'document': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(document)])),
            'tokens': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(tokens)])),
            'sentences': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(sentences)])),
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
        for doc, tok, sent in examples_batch:
            tfrecord_writer.write(create_example(doc, tok, sent))


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset from files with CoNLL-U markup')
    parser.add_argument(
        'src_path',
        type=str,
        help='Directory with source CoNLL-U files')
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
        help='How many times repeat source data (useful due paragraphs shuffling and random glue)')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.src_path) and os.path.isdir(argv.src_path)
    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path)
    assert argv.doc_size > 0
    assert argv.rec_size > 0
    assert argv.num_repeats > 0

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Searching source files in {}'.format(argv.src_path))
    source_files = []
    for root, _, files in os.walk(argv.src_path):
        source_files.extend([os.path.join(root, file) for file in files if file.endswith('.conllu')])
    np.random.shuffle(source_files)
    tf.logging.info('Found {} files. Each will be repeated {} time(s)'.format(len(source_files), argv.num_repeats))

    base_name = os.path.basename(argv.src_path.strip(os.sep))

    examples_queue = []
    for file_name in source_files * argv.num_repeats:
        tf.logging.info('Parsing CoNLL-U file {}'.format(file_name))
        source_paragraphs = parse_conllu(file_name)
        np.random.shuffle(source_paragraphs)
        tf.logging.info('Found {} paragraphs'.format(len(source_paragraphs)))

        while len(source_paragraphs):
            paragraphs_todo, source_paragraphs = source_paragraphs[:argv.rec_size], source_paragraphs[argv.rec_size:]

            tf.logging.info('Augmenting {}K paragraphs'.format(len(paragraphs_todo) // 1000))
            augmented_todo = augment_dataset(paragraphs_todo)

            tf.logging.info('Tokenizing {}K paragraphs'.format(len(augmented_todo) // 1000))
            tokenized_todo = tokenize_dataset(augmented_todo, batch_size=argv.doc_size)

            tf.logging.info('Converting {}K paragraphs'.format(len(tokenized_todo) // 1000))
            examples_todo = make_documents(tokenized_todo, doc_size=argv.doc_size)
            examples_queue.extend(examples_todo)

            if len(examples_queue) >= argv.rec_size:
                tf.logging.info('Saving dataset to {}'.format(argv.dest_path))
                examples_todo, examples_queue = examples_queue[:argv.rec_size], examples_queue[argv.rec_size:]
                write_dataset(argv.dest_path, base_name, examples_todo)

    if len(examples_queue):
        tf.logging.info('Saving dataset to {}'.format(argv.dest_path))
        examples_todo, examples_queue = examples_queue[:argv.rec_size], examples_queue[argv.rec_size:]
        write_dataset(argv.dest_path, base_name, examples_todo)
