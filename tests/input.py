# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfdsbd.input import train_input_fn, predict_input_fn


class TestTrainInputFn(tf.test.TestCase):
    def testNormal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', 'train*.tfrecords.gz')
        batch_size = 2

        dataset = train_input_fn(wildcard, batch_size)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        with self.test_session() as sess:
            features, labels = sess.run(features)
            self.assertEqual(type(features), dict)
            self.assertEqual(sorted(features.keys()), ['document', 'length', 'tokens'])
            self.assertEqual(len(features['document']), batch_size)
            self.assertEqual(list(features['length']), [92, 112])
            self.assertEqual([len(_) for _ in labels], [112, 112])


class TestPredictInputFn(tf.test.TestCase):
    def testNormal(self):
        expected = [
            u'Интернет', u'-', u'ресурсы', u' ', u'netBridge', u' ', u'изначально', u' ', u'строились', u' ', u'как',
            u' ', u'копии', u' ', u'самых', u' ', u'известных', u' ', u'и', u' ', u'популярных', u' ', u'американских',
            u' ', u'сайтов', u' ', u'и', u',', u' ', u'очевидно', u',', u' ', u'предназначались', u' ', u'на', u' ',
            u'продажу', u' ', u'инвестору', u'.',
            u' ',
            u'С', u'.', u' ', u'Старостин', u':', u' ', u'-', u'-', u' ', u'15', u' ', u'тысяч', u' ', u'лет', u' ',
            u'-', u'-', u' ', u'это', u' ', u'время', u' ', u'существования', u' ', u'ностратической', u' ', u'семьи',
            u',', u' ', u'древней', u' ', u'языковой', u' ', u'общности', u',', u' ', u'которая', u' ', u'позднее',
            u' ', u'породила', u' ', u'индоевропейские', u',', u' ', u'алтайские', u',', u' ', u'уральские', u' ', u'и',
            u' ', u'некоторые', u' ', u'другие', u' ', u'языки', u'.']

        dataset = predict_input_fn([u''.join(expected)])
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        with self.test_session() as sess:
            result = sess.run(features)
            self.assertEqual(type(result), dict)
            self.assertEqual(sorted(result.keys()), ['document', 'length', 'tokens'])
            self.assertEqual(len(result['document']), 1)
            self.assertEqual(result['length'], 99)


if __name__ == "__main__":
    tf.test.main()
