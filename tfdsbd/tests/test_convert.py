# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile
import unittest

from ..convert import tokenize_dataset, extract_vocab, make_dataset, write_dataset


class TestTokenizeDataset(unittest.TestCase):
    def testNormal(self):
        source = 'Single sentence.\n\nFirst sentence in paragraph.\r\nSecond sentence in paragraph.'
        expected = [
            [
                [u'Single', u' ', u'sentence', u'.'],
            ],
            [
                [u'First', u' ', u'sentence', u' ', u'in', u' ', u'paragraph', u'.'],
                [u'Second', u' ', u'sentence', u' ', u'in', u' ', u'paragraph', u'.'],
            ],
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = u'Single sentence.\n  \n\t\nNext single sentence.'
        expected = [
            [
                [u'Single', u' ', u'sentence', u'.'],
                [u' ', u' '],
                [u'\t'],
                [u'Next', u' ', u'single', u' ', u'sentence', u'.']
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)


class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def testDocNoDot(self):
        source = [
            [
                ['First', ' ', 'sentence', ' ', 'without', ' ', 'dot'],
                ['Second', ' ', '"', 'sentence', '!', '"'],
                ['Third', ' ', 'sentence', '.'],
            ],
            [
                [u'Первое', ' ', u'предложение', ' ', u'в', ' ', u'параграфе'],
                ['Second', ' ', 'sentence', ' ', 'in', ' ', 'paragraph', '.'],
            ],
        ]

        expected_documents = [
            'First         sentence   without        dot   ' +
            'Second  "sentence!"     ' +
            'Third sentence.',

            u'Первое предложение   в    параграфе ' +
            'Second    sentence   in  paragraph.',

            'First sentence  without     dot\n' +
            'Second  "sentence!"  ' +
            'Third    sentence. ',

            u'Первое предложение в        параграфе\n\n' +
            'Second sentence  in  paragraph.  '
        ]
        expected_labels = [
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 0, 0
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 1
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
            ]
        ]

        result = make_dataset(source, doc_size=2, num_repeats=2)
        result_documents, result_labels = zip(*result)
        result_documents, result_labels = list(result_documents), list(result_labels)

        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_labels, result_labels)


class TestExtractVocab(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def testNormal(self):
        source = make_dataset([
            [
                [u'Single', u' ', u'sentence', u'.'],
            ],
            [
                [u'First', u' ', u'sentence', u' ', u'in', u' ', u'paragraph', u'.'],
                [u'Second', u' ', u'sentence', u' ', u'in', u' ', u'paragraph', u'.'],
            ],
        ], doc_size=2, num_repeats=1)
        expected = ['< >', '<se', '<.>', '<sen', '<sent', '<sente', '<sentence>', 'ce>', 'ence>', 'nce>', 'tence>']
        result = extract_vocab(source, 3, 6, 3)
        self.assertEqual(expected, result.items())

    def testNewlines(self):
        source = make_dataset([
            [
                [u'Single', ' ', u'sentence'],
            ],
            [
                [u'First', ' ', u'sentence', ' ', u'in', ' ', u'paragraph'],
                [u'Second', ' ', u'sentence', ' ', u'in', ' ', u'paragraph'],
            ],
        ], doc_size=3, num_repeats=2)
        result = extract_vocab(source, 3, 6, 2)
        self.assertTrue('<\n>' in result.items())


class TestWriteDataset(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = [
            (
                u'First sentence without  dot\n' +
                u'Second "sentence!" ' +
                u'Third sentence.\t' +
                u'Первое предложение в  параграфе\n' +
                u'Second sentence in  paragraph.',

                [
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            (
                u'First  sentence without dot ' +
                u'Second "sentence!" ' +
                u'Third  sentence. ' +
                u'Первое предложение в  параграфе\n' +
                u'Second sentence in paragraph.',

                [
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0
                ]
            )
        ]

        write_dataset(self.temp_dir, 'test', 'buffer', 100, source)

        def _parse_function(example_proto):
            print(example_proto)
            features = tf.parse_single_example(
                example_proto,
                features={
                    'document': tf.FixedLenFeature(1, tf.string),
                    'labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                })
            return features['document'][0], features['labels']

        expected_filename = os.path.join(self.temp_dir, '*.tfrecords.gz')
        files = tf.data.TFRecordDataset.list_files(expected_filename)
        dataset = files.interleave(
            lambda f: tf.data.TFRecordDataset([f], compression_type='GZIP'),
            cycle_length=5
        )
        dataset = dataset.map(_parse_function)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with self.test_session() as sess:
            document, labels = sess.run(next_element)
            self.assertEqual(source[0][0].encode('utf-8'), document)
            self.assertEqual(source[0][1], list(labels))

            document, labels = sess.run(next_element)
            self.assertEqual(source[1][0].encode('utf-8'), document)
            self.assertEqual(source[1][1], list(labels))
