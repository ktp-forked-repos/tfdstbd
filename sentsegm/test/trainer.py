# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import unittest
from ..dataset import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        data_params = {
            'data_dir': 'test/data/',
            'batch_size': 2,
            'test_size': 0.1,
            'doc_size': 3,
            'random_seed': 41
        }

        self.trainer = Trainer(**data_params)
        tf.set_random_seed(data_params['random_seed'])

    def test_sentence_loader(self):
        self.trainer._load_sentences()

        self.assertIsInstance(self.trainer._full_dataset, list)
        self.assertEqual(len(self.trainer._full_dataset), 15)

        self.assertEqual(len(self.trainer._full_dataset[0]), 1)
        self.assertEqual(len(self.trainer._full_dataset[0][0]), 30)

        self.assertEqual(len(self.trainer._full_dataset[13]), 2)
        self.assertEqual(self.trainer._full_dataset[13][1], [u'Дальше', u' ', u'больше', u'.'])

    def test_data_generator(self):
        source = [
            [  # paragraph with 1 sentence
                ['Hello', ' ', 'everyone', '.']  # Hello everyone.
            ],
            [  # paragraph with 2 sentences
                ['"', 'Hello', ' ', 'everyone', '.'],  # "Hello everyone.
                ['-', ' ', 'said', ' ', 'Mike', '.', '"']  # - said Mike."
            ],
        ]
        expected = [  # 3 sentence boundaries (True)
            ('Hello', False), (' ', False), ('everyone', False), ('.', False), (' ', True),

            ('"', False), ('Hello', False), (' ', False), ('everyone', False), ('.', False), (' ', True),

            ('-', False), (' ', False), ('said', False), (' ', False), ('Mike', False), ('.', False), ('"', False),
            (' ', True)
        ]
        res_X, res_y = next(self.trainer._data_generator(source))
        res = list(zip(res_X, res_y))
        self.assertEqual(res, expected)

    def test_real_generators(self):
        test = list(self.trainer._test_generator())
        self.assertGreater(len(test), 0)

        train = list(self.trainer._train_generator())
        self.assertGreater(len(train), 1)

    def test_train_dataset(self):
        iterator = self.trainer.train_dataset().make_initializable_iterator()
        X, y = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            eX, ey = sess.run([X, y])

        self.assertEqual(len(ey), 2)
        self.assertGreater(len(ey[0]), 1)

        self.assertEqual(len(eX), 2)
        self.assertGreater(len(eX[0]), 1)

        self.assertTrue(eX[0][-1] == 0 or eX[1][-1] == 0)


if __name__ == "__main__":
    tf.test.main()
