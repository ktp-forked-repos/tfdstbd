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

from ..prepare import tokenize_dataset, extract_vocab, make_dataset, write_dataset
from ..input import train_input_fn


class TestTokenizeDataset(unittest.TestCase):
    def testNormal(self):
        source = b'Single sentence.\n\nFirst sentence in paragraph.\r\nSecond sentence in paragraph.'
        expected = [
            [
                [b'Single', b' ', b'sentence', b'.'],
            ],
            [
                [b'First', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
                [b'Second', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
            ],
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = b'Single sentence.\n  \n\t\nNext single sentence.'
        expected = [
            [
                [b'Single', b' ', b'sentence', b'.'],
                [b' ', b' '],
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
            b'First sentence       without    dot  \n\t' +
            b'Second   "sentence!" ' +
            b'Third sentence.\t  \t ',

            b'First  sentence without dot\n\n\t\n\n\n\n\n' +
            b'Second "sentence!" ' +
            b'Third     sentence.  ',

            u'Первое предложение  в параграфе\t\t'.encode('utf-8') +
            b'Second   sentence in         paragraph. ',

            u'Первое  предложение  в параграфе  '.encode('utf-8') +
            b'Second      sentence  in paragraph. ',
        ]
        expected_labels = [
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 1, 1, 1, 1, 1
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
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
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
            (
                b'First  sentence without dot ' +
                b'Second "sentence!" ' +
                b'Third  sentence. ' +
                u'Первое предложение в  параграфе\n'.encode('utf-8') +
                u'Second sentence in paragraph.'.encode('utf-8'),

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

        wildcard = os.path.join(self.temp_dir, '*.tfrecords.gz')
        dataset = train_input_fn(wildcard, 1)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with self.test_session() as sess:
            features, labels = sess.run(next_element)
            self.assertEqual(source[0][0], features['documents'][0])
            self.assertEqual(source[0][1], labels[0].tolist())

            features, labels = sess.run(next_element)
            self.assertEqual(source[1][0], features['documents'][0])
            self.assertEqual(source[1][1], labels[0].tolist())


class TestExtractVocab(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = make_dataset([
            [
                [b'Single', b' ', b'sentence', b'.'],
            ],
            [
                [b'First', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
                [b'Second', b' ', b'sentence', b' ', b'in', b' ', b'paragraph', b'.'],
            ],
        ], doc_size=2, num_repeats=1)
        write_dataset(self.temp_dir, 'test', 'buffer', 100, source)

        expected = [b'< >', b'<se', b'<.>', b'<sen', b'<sent', b'<sente', b'<sentence>', b'ce>', b'ence>', b'nce>',
                    b'tence>']
        result = extract_vocab(self.temp_dir, 'test', 'buffer', 3, 6, 3)
        self.assertEqual(expected, result.items())

    def testNewlines(self):
        source = make_dataset([
            [
                [b'Single', ' ', b'sentence'],
            ],
            [
                [b'First', ' ', b'sentence', ' ', b'in', ' ', b'paragraph'],
                [b'Second', ' ', b'sentence', ' ', b'in', ' ', b'paragraph'],
            ],
        ], doc_size=3, num_repeats=2)
        write_dataset(self.temp_dir, 'test', 'buffer', 100, source)

        result = extract_vocab(self.temp_dir, 'test', 'buffer', 3, 6, 2)
        self.assertTrue(b'<\n>' in result.items())


