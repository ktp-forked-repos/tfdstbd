# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..transform import extract_features, extract_ngrams


class TestExtractFeatures(tf.test.TestCase):
    def testFeatures(self):
        source = ['123', u'нижний', u'ВЕРХНИЙ', 'Предложение']
        expected = [
            [1., 0., 0., 0., 0.11920291930437088],
            [0., 1., 0., 0., 0.7310585975646973],
            [0., 0., 1., 0., 0.8807970285415649],
            [0., 0., 0., 1., 0.9975274205207825],
        ]
        result = extract_features(source)

        with self.test_session() as sess:
            features = sess.run(result)
            self.assertEqual(expected, features.tolist())


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
