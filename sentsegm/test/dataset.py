# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import unittest
from ..dataset import Trainer
from ..ops import split_tokens


class TestTrainer(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(__file__)
        data_params = {
            'data_dir': os.path.join(base_dir, 'data'),
            'batch_size': 2,
            'test_size': 0.1,
            'doc_size': 3,
            'random_seed': 41
        }

        tf.set_random_seed(data_params['random_seed'])
        self.trainer = Trainer(**data_params)

    def test_sentence_loader(self):
        self.trainer._load_dataset()

        self.assertIsInstance(self.trainer._full_data, list)
        self.assertEqual(len(self.trainer._full_data), 15)

        self.assertEqual(len(self.trainer._full_data[0]), 3)
        self.assertEqual(len(self.trainer._full_data[0][0]), 30)

        self.assertEqual(len(self.trainer._full_data[1]), 1)

    def test_data_generator(self):
        source = [
            [  # paragraph with 1 sentence
                ['Hello', ' ', 'everyone', '.'],
            ],
            [  # paragraph with 2 sentences
                ['"', 'Hello', ' ', 'everyone', '.'],
                ['-', ' ', 'said', ' ', 'Mike', '.', '"'],
            ],
        ]
        expected_X = 'Hello everyone. "Hello everyone. - said Mike."'
        expected_y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        res_X, res_y = next(self.trainer._data_generator(source))

        self.assertEqual(expected_X, res_X)
        self.assertEqual(expected_y, res_y)

    def test_real_generators(self):
        test = list(self.trainer._test_generator())
        self.assertGreater(len(test), 0)

        train = list(self.trainer._train_generator())
        self.assertGreater(len(train), 1)

    def test_train_dataset(self):
        expected_X = u'Интернет-ресурсы netBridge изначально строились как копии самых известных и популярных американских сайтов и, очевидно, предназначались на продажу инвестору. ' + \
                     u'С. Старостин: -- 15 тысяч лет -- это время существования ностратической семьи, древней языковой общности, которая позднее породила индоевропейские, алтайские, уральские и некоторые другие языки.'
        expected_y = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
            1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]

        iterator = self.trainer.train_dataset().make_initializable_iterator()
        X, y = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            sess.run([X, y])
            eX, ey = sess.run([X, y])

        self.assertEqual(expected_X, eX[0].decode('utf-8'))
        self.assertListEqual(expected_y, list(ey[0]))


if __name__ == "__main__":
    tf.test.main()
