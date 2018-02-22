# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import tensorflow as tf
import unittest
from sentsegm.input import Trainer, predict_input_fn


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        base_dir = os.path.join(os.path.dirname(__file__), 'data')
        shutil.copyfile(os.path.join(base_dir, 'dataset.txt'), os.path.join(self.temp_dir, 'dataset.txt'))

        data_params = {
            'data_dir': self.temp_dir,
            'batch_size': 2,
            'test_size': 0.1,
            'doc_size': 3,
        }
        self.trainer = Trainer(**data_params)

        random_seed = 41
        self.trainer._random_generator.seed(random_seed)
        tf.set_random_seed(random_seed)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testSentenceLoader(self):
        self.trainer._load_dataset()

        self.assertIsInstance(self.trainer._full_data, list)
        self.assertEqual(len(self.trainer._full_data), 15)

        self.assertEqual(len(self.trainer._full_data[0]), 3)
        self.assertEqual(len(self.trainer._full_data[0][0]), 30)

        self.assertEqual(len(self.trainer._full_data[1]), 1)

    def testDataGenerator(self):
        source = [
            [  # paragraph with 1 sentence
                ['Hello', ' ', 'everyone', '.'],
            ],
            [  # paragraph with 2 sentences
                ['"', 'Hello', ' ', 'everyone', '.'],
                ['-', ' ', 'said', ' ', 'Mike', '.', '"'],
            ],
        ]
        expected_X = ['Hello', ' ', 'everyone', '.', ' ', '"', 'Hello', ' ', 'everyone', '.', ' ', '-', ' ', 'said',
                      ' ', 'Mike', '.', '"']
        expected_X = [_.encode('utf-8') for _ in expected_X]  # required due to TF 1.6.0rc1 bug in Python2
        expected_y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        result = next(self.trainer._data_generator(source))

        self.assertEqual((
            {'document': expected_X},
            expected_y
        ), result)

    def testTrainInputFn(self):
        expected_X_py2 = [
            u'Власти', u' ', u'и', u' ', u'эксперты', u' ', u'призывают', u' ', u'к', u' ', u'спокойствию', u' ', u'—',
            u' ', u'ничего', u' ', u'страшного', u' ', u'при', u' ', u'нашем', u' ', u'—', u' ', u'то', u' ',
            u'золотовалютном', u' ', u'запасе', u' ', u'случиться', u' ', u'не', u' ', u'может', u'.',
            u' ',
            u'Однако', u' ', u'технология', u' ', u'извлечения', u' ', u'из', u' ', u'подозрительных', u' ',
            u'бизнесменов', u' ', u'бюджетных', u' ', u'доходов', u',', u' ', u'так', u' ', u'знаменательно', u' ',
            u'отработанная', u' ', u'на', u' ', u'том', u' ', u'же', u' ', u'Джохтаберидзе', u',', u' ', u'остается',
            u' ', u'прежней', u',', u' ', u'пусть', u' ', u'и', u' ', u'не', u' ', u'столь', u' ', u'афишируемой', u'.']
        expected_y_py2 = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected_X_py3 = [
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
        expected_y_py3 = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
            1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if sys.version_info > (3,):  # required due to different random algorithms in Python 2 & 3
            expected_Xd = expected_X_py3
            expected_y = expected_y_py3
            expected_Xl = [99, 31]
        else:
            expected_Xd = expected_X_py2
            expected_y = expected_y_py2
            expected_Xl = [84, 66]
        expected_Xd = [_.encode('utf-8') for _ in expected_Xd]  # required due to TF 1.6.0rc1 bug in Python2

        iterator = self.trainer.train_input_fn().make_initializable_iterator()
        X, y = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            sess.run([X, y])
            res_X, res_y = sess.run([X, y])

            self.assertEqual(dict, type(res_X))
            self.assertEqual(['document', 'length'], sorted(res_X.keys()))

            self.assertEqual(expected_y, list(res_y[0]))
            self.assertEqual(expected_Xd, list(res_X['document'][0]))
            self.assertEqual(len(res_X['document'][0]), len(res_y[0]))

            self.assertEqual(expected_Xl, list(res_X['length']))


class TestPredictInputFn(unittest.TestCase):
    def testInputFn(self):
        document = u'Власти и эксперты призывают к спокойствию — ничего страшного при нашем — то золотовалютном ' \
                   u'запасе случиться не может. Однако технология извлечения из подозрительных бизнесменов бюджетных ' \
                   u'доходов, так знаменательно отработанная на том же Джохтаберидзе, остается прежней, пусть и не ' \
                   u'столь афишируемой.'
        dataset = predict_input_fn([document, 'short'], batch_size=10)
        iterator = dataset.make_initializable_iterator()
        X = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            result = sess.run(X)

        self.assertEqual(1, len(result))
        self.assertEqual(dict, type(result[0]))
        self.assertEqual(['document', 'length'], sorted(result[0].keys()))

        self.assertEqual(len(result[0]['document'][0]), result[0]['length'][0])

if __name__ == "__main__":
    tf.test.main()
