# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..transform import extract_features, extract_ngrams


class TestExtractFeatures(tf.test.TestCase):
    def testInferenceShape(self):
        source = tf.string_split(['a bc def', 'ghij'])
        result = extract_features(source)

        self.assertEqual([None, None, 6], result.shape.as_list())

    def testActualShape(self):
        source = tf.string_split(['a bc def', 'ghij'])
        result = extract_features(source)
        result = tf.shape(result)

        with self.test_session():
            result = result.eval()
            self.assertAllEqual([2, 3, 6], result)

    def testEmpty(self):
        source = tf.SparseTensor(
            indices=[[0, 0]],
            values=[''],
            dense_shape=[1, 1]
        )
        result = extract_features(source)

        with self.test_session():
            result = result.eval()
            print(result)
            self.assertListEqual([[[
                0., 1., 0., 0., 0., 0.,
            ]]], result.tolist())

    def testSmall(self):
        source = tf.string_split(['1'])
        result = extract_features(source)

        with self.test_session():
            result = result.eval()
            print(result)
            self.assertListEqual([[[
                0.06666667014360428, 1., 0., 0., 0., 0.
            ]]], result.tolist())

    def testFeatures(self):
        tokens = tf.string_split([u'123 нижний ВЕРХНИЙ Предложение 34т аC'], delimiter=' ')
        expected = [[
            [0.20000000298023224, 1., 0., 0., 0., 0.],
            [0.4000000059604645, 0., 1., 0., 0., 0.],
            [0.46666666865348816, 0., 0., 1., 0., 0.],
            [0.7333333492279053, 0., 0., 0., 1., 0.],
            [0.20000000298023224, 0., 1., 0., 0., 0.],
            [0.13333334028720856, 0., 0., 0., 0., 1.],
        ]]
        result = extract_features(tokens)
        self.maxDiff = None

        with self.test_session() as sess:
            result = sess.run(result)
            print(result)
            self.assertListEqual(expected, result.tolist())


class TestExtractNgrams(tf.test.TestCase):
    def testNgrams(self):
        source = ['123', u'Регистр', u'км⁴']
        expected = tf.convert_to_tensor([
            ['<0', '00', '00', '0>', '<00', '000', '00>', '<000>', '', '', '', '', '', '', '', ''],
            ['<р', 'ре', 'ег', 'ги', 'ис', 'ст', 'тр', 'р>', '<ре', 'рег', 'еги', 'гис', 'ист', 'стр', 'тр>',
             '<регистр>'],
            ['<к', 'км', 'м0', '0>', '<км', 'км0', 'м0>', '<км0>', '', '', '', '', '', '', '', ''],
        ])
        result = extract_ngrams(source, 2, 3)
        result = tf.sparse_tensor_to_dense(result, default_value='')

        with self.test_session() as sess:
            expected, result = sess.run([expected, result])
            self.assertEqual(expected.tolist(), result.tolist())
