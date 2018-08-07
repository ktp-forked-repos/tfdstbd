# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..input import train_input_fn


class TestTrainInputFn(tf.test.TestCase):
    def testNormal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', 'train*.tfrecords.gz')
        batch_size = 2

        dataset = train_input_fn(wildcard, batch_size, 1, 1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        with self.test_session() as sess:
            features, labels = sess.run(features)

        self.assertEqual(dict, type(features))
        self.assertEqual([
            'documents',
            'length',
            'lower_case',
            'mixed_case',
            'ngrams',
            'no_case',
            'title_case',
            'upper_case',
            'words',
        ], sorted(features.keys()))
        self.assertEqual(batch_size, len(features['documents']))
        self.assertEqual(2, len(labels.dense_shape))
        self.assertEqual(batch_size, labels.dense_shape[0])

        self.assertAllEqual(labels.dense_shape, features['words'].dense_shape)
        self.assertAllEqual(labels.dense_shape, features['length'].dense_shape)


if __name__ == "__main__":
    tf.test.main()
