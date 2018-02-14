# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..ops import split_tokens


class SplitTokensTest(tf.test.TestCase):
    def testEmptyBatch(self):
        result_op = split_tokens([])
        expected_op = tf.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.string),
            dense_shape=[0, 0]
        )

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected.indices, result.indices)
            self.assertAllEqual(expected.values, result.values)
            self.assertAllEqual(expected.dense_shape, result.dense_shape)

    def testEmptyRow(self):
        result_op = split_tokens([''])
        expected_op = tf.SparseTensor(
            indices=[[0, 0]],
            values=[''],
            dense_shape=[1, 1]
        )

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected.indices, result.indices)
            self.assertAllEqual(expected.values, result.values)
            self.assertAllEqual(expected.dense_shape, result.dense_shape)

    def testShape(self):
        result_op = split_tokens(['word', 'word word'])
        expected_op = tf.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1], [1, 2]],
            values=['word', 'word', ' ', 'word'],
            dense_shape=[2, 3]
        )

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected.indices, result.indices)
            self.assertAllEqual(expected.values, result.values)
            self.assertAllEqual(expected.dense_shape, result.dense_shape)

    def testRestore(self):
        source = ['word', 'word word', 'Hey\n\tthere\t«word», !!!']
        result_op = split_tokens(source)
        result_op = tf.sparse_tensor_to_dense(result_op, default_value='')
        result_op = tf.reduce_join(result_op, 1)

        expected_op = tf.constant(source)

        with self.test_session():
            expected, result = expected_op.eval(), result_op.eval()
            self.assertAllEqual(expected, result)

    def testWrappedWord(self):
        result_op = split_tokens([
            ' "word" ',
            u' «word» ',
            u' „word“ ',
            ' {word} ',
            ' (word) ',
            ' [word] ',
            ' <word> ',
        ])
        expected_values = [
            ' ', '"', 'word', '"', ' ',
            ' ', u'«', 'word', u'»', ' ',
            ' ', u'„', 'word', u'“', ' ',
            ' ', '{', 'word', '}', ' ',
            ' ', '(', 'word', ')', ' ',
            ' ', '[', 'word', ']', ' ',
            ' ', '<', 'word', '>', ' '
        ]
        expected_values = [_.encode('utf-8') for _ in expected_values]

        with self.test_session():
            result = result_op.eval()
            self.assertAllEqual(expected_values, result.values)

    def testWordPunkt(self):
        result_op = split_tokens([
            ' word. ',
            ' word.. ',
            ' word... ',
            u' word… '
            ' word, ',
            ' word., ',
            ' word: ',
            ' word; ',
            ' word! ',
            ' word? ',
            ' word% ',
            ' $word ',
        ])
        expected_values = [
            ' ', 'word', '.', ' ',
            ' ', 'word', '.', '.', ' ',
            ' ', 'word', '.', '.', '.', ' ',
            ' ', 'word', u'…', ' ',
            ' ', 'word', ',', ' ',
            ' ', 'word', '.', ',', ' ',
            ' ', 'word', ':', ' ',
            ' ', 'word', ';', ' ',
            ' ', 'word', '!', ' ',
            ' ', 'word', '?', ' ',
            ' ', 'word', '%', ' ',
            ' ', '$', 'word', ' '
        ]
        expected_values = [_.encode('utf-8') for _ in expected_values]

        with self.test_session():
            result = result_op.eval()
            self.assertAllEqual(expected_values, result.values)

    def testComplexWord(self):
        result_op = split_tokens([
            ' test@test.com ',
            ' www.test.com ',
            ' word..word ',
            ' word+word-word ',
            ' word\\word/word#word ',
        ])
        expected_values = [
            ' ', 'test', '@', 'test', '.', 'com', ' ',
            ' ', 'www', '.', 'test', '.', 'com', ' ',
            ' ', 'word', '.', '.', 'word', ' ',
            ' ', 'word', '+', 'word', '-', 'word', ' ',
            ' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '
        ]
        expected_values = [_.encode('utf-8') for _ in expected_values]

        with self.test_session():
            result = result_op.eval()
            self.assertAllEqual(expected_values, result.values)

    def testCharBreaks(self):
        result_op = split_tokens([
            u' а_также ',
            u' т.д. ',
            u' José ',
            u' ЁёЁёй ',
            ' 1,5 ',
            ' 01.01.2018 ',
        ])
        expected_values = [
            ' ', u'а', '_', u'также', ' ',
            ' ', u'т', '.', u'д', '.', ' ',
            ' ', u'José', ' ',
            ' ', u'ЁёЁёй', ' ',
            ' ', '1', ',', '5', ' ',
            ' ', '01', '.', '01', '.', '2018', ' '
        ]
        expected_values = [_.encode('utf-8') for _ in expected_values]

        with self.test_session():
            result = result_op.eval()
            self.assertAllEqual(expected_values, result.values)


if __name__ == "__main__":
    tf.test.main()
