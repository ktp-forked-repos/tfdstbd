# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..ops import split_tokens, extract_features


class ExtractFeaturesTest(tf.test.TestCase):
    features_count = 36
    feature_case_offset = 0
    feature_case_length = 5
    feature_length_offset = 5
    feature_length_length = 2
    feature_chartypes_offset = 7
    feature_chartypes_length = 9
    feature_beginend_offset = 16
    feature_beginend_length = 4
    feature_charsound_offset = 20
    feature_charsound_length = 16

    def testEmptyBatch(self):
        result_op = tf.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.string),
            dense_shape=[0, 0]
        )
        result_op = extract_features(result_op)

        expected_op = tf.SparseTensor(
            indices=tf.zeros([0, 3], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.float32),
            dense_shape=[0, 0, self.features_count]
        )

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected.indices, result.indices)
            self.assertAllEqual(expected.values, result.values)
            self.assertAllEqual(expected.dense_shape, result.dense_shape)

    def testEmptyRow(self):
        result_op = tf.SparseTensor(
            indices=[[0, 0]],
            values=[''],
            dense_shape=[1, 1]
        )
        result_op = extract_features(result_op)

        expected_op = tf.SparseTensor(
            indices=[[0, 0, 0], [0, 0, 5]],
            values=[1., 1.],
            dense_shape=[1, 1, self.features_count]
        )

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected.indices, result.indices)
            self.assertAllEqual(expected.values, result.values)
            self.assertAllEqual(expected.dense_shape, result.dense_shape)

    def testShape(self):
        result_op = split_tokens(['word', 'word word'])
        result_op = extract_features(result_op)
        result_op = tf.sparse_tensor_to_dense(result_op)
        result_op = tf.shape(result_op)

        expected = [2, 3, self.features_count]
        with self.test_session():
            result = result_op.eval()
            self.assertAllEqual(expected, result)

    def testCaseFeatures(self):
        result_op = split_tokens([
            '0',  # no case
            'HELLO',  # upper case
            'hello',  # lower case
            'Hello',  # title case
            'HelLo',  # mixed case
        ])
        result_op = extract_features(result_op)
        result_op = tf.sparse_tensor_to_dense(result_op)
        result_op = tf.slice(result_op, [0, 0, self.feature_case_offset], [-1, -1, self.feature_case_length])

        expected = [
            [[1., 0., 0., 0., 0.]],
            [[0., 1., 0., 0., 0.]],
            [[0., 0., 1., 0., 0.]],
            [[0., 0., 0., 1., 0.]],
            [[0., 0., 0., 0., 1.]]
        ]

        with self.test_session():
            result = result_op.eval()
            self.assertAllEqual(expected, result)

    def testLengthFeatures(self):
        result_op = split_tokens([
            '',  # empty
            '1',  # 1 char
            '12',  # 2 char
            '01234567890123456789',  # 20 char
        ])
        result_op = extract_features(result_op)
        result_op = tf.sparse_tensor_to_dense(result_op)
        result_op = tf.slice(result_op, [0, 0, self.feature_length_offset], [-1, -1, self.feature_length_length])

        expected_op = tf.constant([
            [[1., 0.]],
            [[0., 0.04]],
            [[0., 0.08]],
            [[0., 0.8]]
        ], dtype=tf.float32)

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected, result)

    def testCharTypesFeatures(self):
        result_op = split_tokens([
            '',
            '1',
            'a',
            u'б',
            '_',
            '.',
            '!',
            ')',
            '"',
            ' ',
            u'\u00A0',
            '\t',
            '\n',
        ])
        result_op = extract_features(result_op)
        result_op = tf.sparse_tensor_to_dense(result_op)
        result_op = tf.slice(result_op, [0, 0, self.feature_chartypes_offset], [-1, -1, self.feature_chartypes_length])

        expected_op = tf.constant([
            # digit alpha punct graph blank space cntrl print base
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.]],  # empty
            [[1., 0., 0., 1., 0., 0., 0., 1., 1.]],  # number
            [[0., 1., 0., 1., 0., 0., 0., 1., 1.]],  # latin letter
            [[0., 1., 0., 1., 0., 0., 0., 1., 1.]],  # cyrillic letter
            [[0., 0., 1., 1., 0., 0., 0., 1., 0.]],  # underscore
            [[0., 0., 1., 1., 0., 0., 0., 1., 0.]],  # dot
            [[0., 0., 1., 1., 0., 0., 0., 1., 0.]],  # exclamation
            [[0., 0., 1., 1., 0., 0., 0., 1., 0.]],  # close bracket
            [[0., 0., 1., 1., 0., 0., 0., 1., 0.]],  # quote
            [[0., 0., 0., 0., 1., 1., 0., 1., 0.]],  # space
            [[0., 0., 0., 0., 1., 1., 0., 1., 0.]],  # nobreak space
            [[0., 0., 0., 0., 1., 1., 1., 0., 0.]],  # tab
            [[0., 0., 0., 0., 0., 1., 1., 0., 0.]],  # newline
        ], dtype=tf.float32)

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected, result)

    def testBeginEndFeatures(self):
        result_op = split_tokens([
            '',
            '1C',
            'c1',
            'CC',
            '19',
        ])
        result_op = extract_features(result_op)
        result_op = tf.sparse_tensor_to_dense(result_op)
        result_op = tf.slice(result_op, [0, 0, self.feature_beginend_offset], [-1, -1, self.feature_beginend_length])

        expected_op = tf.constant([
            [[0., 0., 0., 0.]],
            [[0., 1., 1., 0.]],
            [[1., 0., 0., 1.]],
            [[1., 0., 1., 0.]],
            [[0., 1., 0., 1.]],
        ], dtype=tf.float32)

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected, result)

    def testCharSoundFeatures(self):
        result_op = split_tokens([
            '',
            u'мой',
            u'англ',
            u'ночь',
            'mrs',
            'ny',
            'yes',
        ])
        result_op = extract_features(result_op)
        result_op = tf.sparse_tensor_to_dense(result_op)
        result_op = tf.slice(result_op, [0, 0, self.feature_charsound_offset], [-1, -1, self.feature_charsound_length])

        expected_op = tf.constant([
            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ]],
            [[0.08, 0.08, 0., 0.08, 0.33333334, 0.33333334, 0., 0.33333334, 0., 0., 0., 0.2, 0., 0., 0., 0.33333334]],
            [[0.24, 0.08, 0., 0., 0.75, 0.25, 0., 0., 0.6, 0., 0., 0., 0.75, 0., 0., 0., ]],
            [[0.16, 0.08, 0.08, 0., 0.5, 0.25, 0.25, 0., 0., 0., 0.2, 0., 0., 0., 0.25, 0., ]],
            [[0.24, 0., 0., 0., 1., 0., 0., 0., 0.6, 0., 0., 0., 1., 0., 0., 0., ]],
            [[0.08, 0., 0., 0.08, 0.5, 0., 0., 0.5, 0., 0., 0., 0.2, 0., 0., 0., 0.5, ]],
            [[0.08, 0.08, 0., 0.08, 0.33333334, 0.33333334, 0., 0.33333334, 0.2, 0., 0., 0., 0.33333334, 0., 0., 0., ]],
        ], dtype=tf.float32)

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected, result)


if __name__ == "__main__":
    tf.test.main()
