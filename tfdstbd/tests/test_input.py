# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..input import train_input_fn


class TestTrainInputFn(tf.test.TestCase):
    def testNormal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', '*.tfrecords.gz')
        batch_size = 2

        dataset = train_input_fn(wildcard, batch_size, 1, 1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        with self.test_session() as sess:
            features, labels = sess.run(features)

        self.assertEqual(dict, type(features))
        self.assertEqual([
            'document',
            'is_lower_case',
            'is_mixed_case',
            'is_no_case',
            'is_title_case',
            'is_upper_case',
            'ngrams',
            'word_length',
            'words',
        ], sorted(features.keys()))
        self.assertEqual(batch_size, len(features['document']))

        self.assertEqual(dict, type(labels))
        self.assertEqual(['sentences', 'tokens'], sorted(labels.keys()))

        self.assertEqual(2, len(labels['tokens'].shape))
        self.assertEqual(batch_size, labels['sentences'].shape[0])

        self.assertAllEqual(labels['tokens'].shape, features['words'].shape)
        self.assertAllEqual(labels['sentences'].shape, features['word_length'].dense_shape)


if __name__ == "__main__":
    tf.test.main()
