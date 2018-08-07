# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ..feature import extract_case_length_features, extract_ngram_features


class TestExtractCaseLengthFeatures(tf.test.TestCase):
    def testFeatures(self):
        source = tf.string_split([u'123 нижний ВЕРХНИЙ Предложение 34т аC'])
        word_length, no_case, lower_case, upper_case, title_case, mixed_case = extract_case_length_features(source)

        expected_word_length = [[0.20, 0.40, 0.47, 0.73, 0.20, 0.13]]
        expected_no_case = [[1., 0., 0., 0., 0., 0.]]
        expected_lower_case = [[0., 1., 0., 0., 1., 0.]]
        expected_upper_case = [[0., 0., 1., 0., 0., 0.]]
        expected_title_case = [[0., 0., 0., 1., 1., 0.]]
        expected_mixed_case = [[0., 0., 0., 0., 0., 1.]]

        with self.test_session():
            word_length_value = tf.sparse_tensor_to_dense(word_length, -1.0).eval()
            self.assertAllClose(expected_word_length, word_length_value.tolist(), rtol=1e-2, atol=1e-2)

            no_case_value = tf.sparse_tensor_to_dense(no_case, -1.0).eval()
            self.assertAllEqual(expected_no_case, no_case_value.tolist())

            lower_case_value = tf.sparse_tensor_to_dense(lower_case, -1.0).eval()
            self.assertAllEqual(expected_lower_case, lower_case_value.tolist())

            upper_case_value = tf.sparse_tensor_to_dense(upper_case, -1.0).eval()
            self.assertAllEqual(expected_upper_case, upper_case_value.tolist())

            title_case_value = tf.sparse_tensor_to_dense(title_case, -1.0).eval()
            self.assertAllEqual(expected_title_case, title_case_value.tolist())

            mixed_case_value = tf.sparse_tensor_to_dense(mixed_case, -1.0).eval()
            self.assertAllEqual(expected_mixed_case, mixed_case_value.tolist())


class TestExtractNgramFeatures(tf.test.TestCase):
    def testFeatures(self):
        source = tf.string_split([u'123 Тест тест'])
        ngrams = extract_ngram_features(source, 5, 5)

        expected_ngrams = tf.convert_to_tensor([
            u'<000>',
            u'<тест',
            u'тест>',
            u'<тест',
            u'тест>',
        ])

        with self.test_session():
            expected_value = expected_ngrams.eval()
            ngrams_value = ngrams.eval()
            self.assertAllEqual(expected_value.tolist(), ngrams_value.values.tolist())
