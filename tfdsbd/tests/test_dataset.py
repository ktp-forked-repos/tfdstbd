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

from ..dataset import tokenize_dataset, make_dataset, write_dataset
from ..input import train_input_fn


class TestTokenizeDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def testNormal(self):
        source = b'Single sentence.\n\nFirst sentence in paragraph.\r\nSecond sentence in paragraph.'
        expected = [
            [
                [b'First', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
                [b'Second', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
            ],
            [
                [b'Single', b' ', b'sentence', b'.'],
            ],
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = b'Single sentence.\n  \n\t\nNext single sentence.'
        expected = [
            [
                [b'Single', b' ', b'sentence', b'.'],
                [b'  '],
                [b'\t'],
                [b'Next', b' ', b'single', b' ', b'sentence', b'.']
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testEmpty(self):
        source = b''
        result = tokenize_dataset(source)
        self.assertEqual([], result)


class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def testDocNoDot(self):
        source = [
            [
                [b'First', b' ', b'sentence', b' ', b'without', b' ', b'dot'],
                [b'Second', b' ', b'"', b'sentence', b'!', b'"'],
                [b'Third', b' ', b'sentence', b'.'],
            ],
            [
                [w.encode('utf-8') for w in [u'Первое', ' ', u'предложение', ' ', u'в', ' ', u'параграфе']],
                [b'Second', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
            ],
        ]

        expected_documents = [
            b'First sentence\twithout dot\n\n' +
            b'Second "sentence!" ' +
            b'Third sentence. ',

            b'First sentence without  dot ' +
            b'Second  "sentence!" ' +
            b'Third sentence. ',

            u'Первое предложение в параграфе\n '.encode('utf-8') +
            b'Second sentence  in paragraph. ',

            u'Первое предложение в параграфе\n\n'.encode('utf-8') +
            b'Second sentence in paragraph.   ',
        ]
        expected_labels = [
            [
                b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B', b'B',
                b'N', b'N', b'N', b'N', b'N', b'N', b'B', b'N', b'N', b'N', b'N', b'B'
            ],
            [
                b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                b'N', b'N', b'N', b'N', b'B'
            ],
            [
                b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B', b'B',
                b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B'
            ],
            [
                b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B', b'B',
                b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B'
            ],
        ]

        result = make_dataset(source, doc_size=2, num_repeats=2)
        result_documents, result_labels = zip(*result)
        result_documents, result_labels = list(result_documents), list(result_labels)

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
                b'First sentence without  dot\n' +
                b'Second "sentence!" ' +
                b'Third sentence.\t' +
                u'Первое предложение в  параграфе\n'.encode('utf-8') +
                u'Second sentence in  paragraph.'.encode('utf-8'),

                [
                    b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N'
                ]
            ),
            (
                b'First  sentence without dot ' +
                b'Second "sentence!" ' +
                b'Third  sentence. ' +
                u'Первое предложение в  параграфе\n'.encode('utf-8') +
                u'Second sentence in paragraph.'.encode('utf-8'),

                [
                    b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'B',
                    b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N'
                ]
            )
        ]

        write_dataset(self.temp_dir, 'test', 'buffer', 100, source)

        wildcard = os.path.join(self.temp_dir, '*.tfrecords.gz')
        dataset = train_input_fn(wildcard, 1, 1, 1)
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        document = features['documents']
        labels = tf.sparse_tensor_to_dense(labels, default_value='')

        with self.test_session() as sess:
            document_value, labels_value = sess.run([document, labels])
            self.assertEqual(source[0][0], document_value[0])
            self.assertEqual(source[0][1], labels_value[0].tolist())

            document_value, labels_value = sess.run([document, labels])
            self.assertEqual(source[1][0], document_value[0])
            self.assertEqual(source[1][1], labels_value[0].tolist())
