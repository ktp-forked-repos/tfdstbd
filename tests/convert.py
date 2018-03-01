# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import Random
import os
import shutil
import sys
import tensorflow as tf
import tempfile
import unittest
from tfdsbd.convert import tokenize_dataset, make_dataset, write_dataset

if sys.version_info.major == 3:
    from unittest import mock
else:
    import mock


class TestTokenizeDataset(unittest.TestCase):
    def testNormal(self):
        source = 'Single sentence.\n\nFirst sentence in paragraph.\r\nSecond sentence in paragraph.'
        expected = [
            [
                ['Single', ' ', 'sentence', '.'],
            ],
            [
                ['First', ' ', 'sentence', ' ', 'in', ' ', 'paragraph', '.'],
                ['Second', ' ', 'sentence', ' ', 'in', ' ', 'paragraph', '.'],
            ],
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        source = 'Single sentence.\n  \n\t\nNext single sentence.'
        expected = [
            [
                ['Single', ' ', 'sentence', '.'],
                [' ', ' '],
                ['\t'],
                ['Next', ' ', 'single', ' ', 'sentence', '.']
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)


class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        self.random = Random(0)

    @mock.patch('tfdsbd.convert.random')
    def testDocNoDot(self, random):
        random.randint._mock_side_effect = self.random.randint
        random.shuffle._mock_side_effect = self.random.shuffle
        random.expovariate._mock_side_effect = self.random.expovariate

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
            u'First  sentence without dot ' +
            u'Second "sentence!" ' +
            u'Third  sentence. ' +
            u'Первое предложение в  параграфе\n' +
            u'Second sentence\nin paragraph.',

            u'First sentence without   dot\t' +
            u'Second "sentence!" ' +
            u'Third sentence. ' +
            u'Первое  предложение в параграфе\n' +
            u'Second sentence in paragraph.'
        ]
        expected_labels = [
            [
                0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0
            ]
        ]

        if sys.version_info.major == 3:
            expected_documents = [
                u'First sentence without  dot\n' +
                u'Second "sentence!" ' +
                u'Third sentence.\t' +
                u'Первое предложение в  параграфе\n' +
                u'Second sentence in  paragraph.',

                u'First  sentence without dot ' +
                u'Second "sentence!" ' +
                u'Third  sentence. ' +
                u'Первое предложение в  параграфе\n' +
                u'Second sentence in paragraph.'
            ]
            expected_labels = [
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 0
                ],
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0
                ]
            ]
        expected_documents = [_.encode('utf-8') for _ in expected_documents]

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
            # context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            #     serialized=example_proto,
            #     context_features={'document': tf.FixedLenFeature([], dtype=tf.string)},
            #     sequence_features={'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64)}
            # )
            # return context_parsed['document'], sequence_parsed['labels']

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


if __name__ == "__main__":
    unittest.main()
