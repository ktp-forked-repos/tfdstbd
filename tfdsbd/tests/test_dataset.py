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

from ..dataset import parse_dataset, augment_dataset, tokenize_dataset, make_dataset, write_dataset
from ..input import train_input_fn


class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def testEmpty(self):
        result = parse_dataset(u'')
        self.assertEqual([], result)

    def testNormal(self):
        source = u'Single sentence.\n\nFirst sentence in paragraph.\r\nSecond sentence in paragraph.'
        expected = [
            [u'First sentence in paragraph.', u'Second sentence in paragraph.'],
            [u'Single sentence.'],
        ]
        result = parse_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = u'Single sentence \n  \n\t\nNext single sentence  '
        expected = [[u'Single sentence', u'Next single sentence']]
        result = parse_dataset(source)
        self.assertEqual(expected, result)


class TestAugmentDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(6)

    def testEmpty(self):
        result = augment_dataset([], 1)
        self.assertEqual([], result)

    def testNormal(self):
        source = [
            [u'First sentence in paragraph.', u'Second sentence in paragraph.'],
            [u'Single sentence.'],
        ]
        expected = [
            [u'First sentence in paragraph.\t', u'Second sentence in  paragraph.\n'],
            [u'Single sentence.\t'],
        ]
        result = augment_dataset(source, 100)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = [[u'Single sentence', u'Next single sentence']]
        expected = [[u'Single sentence ', u'Next single sentence ']]
        result = augment_dataset(source, 100)
        self.assertEqual(expected, result)


class TestTokenizeDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def testEmpty(self):
        source = []
        result = tokenize_dataset(source)
        self.assertEqual([], result)

    def testNormal(self):
        source = [
            [u'First sentence in paragraph. \t', u'Second   sentence in paragraph. '],
            [u'Single sentence. '],
        ]
        expected = [
            [
                [u'First', u' ', u'sentence', u' ', u'in', u' ', u'paragraph', u'.', u' ', u'\t'],
                [u'Second', u'   ', u'sentence', u' ', u'in', u' ', u'paragraph', u'.', u' ']
            ],
            [
                [u'Single', u' ', u'sentence', u'.', u' ']
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = [[u'Single sentence \r\n', u'Next single sentence ']]
        expected = [
            [
                [u'Single', u' ', u'sentence', u' ', u'\r\n'],
                [u'Next', u' ', u'single', u' ', u'sentence', u' ']
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)


class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(4)

    def testEmpty(self):
        source = []
        result = make_dataset(source, 2)
        self.assertEqual([], result)

    def testNormal(self):
        source = [
            [
                ['Single', ' ', 'sentence', '.', '\r\n'],
            ],
            [
                ['Another', ' ', 'sentence', '.', '\n'],
            ],
            [
                ['First', ' ', 'sentence', ' ', 'in', ' ', 'paragraph', '.', ' ', '\t'],
                ['Second', '   ', 'sentence', ' ', 'in', ' ', 'paragraph', '.', ' '],
            ],
        ]

        expected_documents = [
            'First sentence in paragraph. \tSecond   sentence in paragraph. ',
            'Another sentence.\nSingle sentence.\r\n',
        ]

        expected_labels = [
            'N,N,N,N,N,N,N,N,B,B,N,N,N,N,N,N,N,N,B',
            'N,N,N,N,B,N,N,N,N,B',
        ]

        result = make_dataset(source, doc_size=15)
        result_documents, result_labels = zip(*result)
        result_documents, result_labels = list(result_documents), list(result_labels)

        print(result_documents)
        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_labels, result_labels)


class TestWriteDataset(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = [
            (
                'First sentence in paragraph. \tSecond   sentence in paragraph. ',
                'N,N,N,N,N,N,N,N,B,B,N,N,N,N,N,N,N,N,B',
            ),
            (
                'Another sentence.\nSingle sentence.\r\n',
                'N,N,N,N,B,N,N,N,N,B',
            )
        ]

        write_dataset(self.temp_dir, 'buffer', source)

        wildcard = os.path.join(self.temp_dir, '*.tfrecords.gz')
        dataset = train_input_fn(wildcard, 1, 1, 1)
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        document = features['documents']

        with self.test_session() as sess:
            document_value, labels_value = sess.run([document, labels])
            self.assertEqual(source[0][0], document_value[0].decode('utf-8'))
            self.assertEqual(source[0][1], b','.join(labels_value[0]).decode('utf-8'))

            document_value, labels_value = sess.run([document, labels])
            self.assertEqual(source[1][0], document_value[0].decode('utf-8'))
            self.assertEqual(source[1][1], b','.join(labels_value[0]).decode('utf-8'))
