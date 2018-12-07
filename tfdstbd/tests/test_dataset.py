# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile
import unittest
from ..dataset import parse_conllu, random_glue, augment_dataset, tokenize_sentence, tokenize_dataset, make_documents, \
    write_dataset
from ..input import train_input_fn


class TestParseConllu(unittest.TestCase):
    def testParseParagraphs(self):
        result = parse_conllu(os.path.join(os.path.dirname(__file__), 'data', 'dataset_paragraphs.conllu'))
        self.assertEqual([
            [
                [u'«', u'Если', u' ', u'передача', u' ', u'цифровых', u' ', u'технологий', u' ', u'сегодня', u' ', u'в',
                 u' ', u'США', u' ', u'происходит', u' ', u'впервые', u',', u' ', u'то', u' ', u'о', u' ', u'мирной',
                 u' ', u'передаче', u' ', u'власти', u' ', u'такого', u' ', u'не', u' ', u'скажешь', u'»', u',', u' ',
                 u'–', u' ', u'написала', u' ', u'Кори', u' ', u'Шульман', u',', u' ', u'специальный', u' ',
                 u'помощник', u' ', u'президента', u' ', u'Обамы', u' ', u'в', u' ', u'своем', u' ', u'блоге', u' ',
                 u'в', u' ', u'понедельник', u'.'],

                [u'Для', u' ', u'тех', u',', u' ', u'кто', u' ', u'следит', u' ', u'за', u' ', u'передачей', u' ',
                 u'всех', u' ', u'материалов', u',', u' ', u'появившихся', u' ', u'в', u' ', u'социальных', u' ',
                 u'сетях', u' ', u'о', u' ', u'Конгрессе', u',', u' ', u'это', u' ', u'будет', u' ', u'происходить',
                 u' ', u'несколько', u' ', u'по-другому', u'.'],
            ],
            [
                [u'Но', u' ', u'в', u' ', u'отступлении', u' ', u'от', u' ', u'риторики', u' ', u'прошлого', u' ', u'о',
                 u' ', u'сокращении', u' ', u'иммиграции', u' ', u'кандидат', u' ', u'Республиканской', u' ', u'партии',
                 u' ', u'заявил', u',', u' ', u'что', u' ', u'в', u' ', u'качестве', u' ', u'президента', u' ', u'он',
                 u' ', u'позволил', u' ', u'бы', u' ', u'въезд', u' ', u'«', u'огромного', u' ', u'количества', u'»',
                 u' ', u'легальных', u' ', u'мигрантов', u' ', u'на', u' ', u'основе', u' ', u'«', u'системы', u' ',
                 u'заслуг', u'»', u'.'],

            ],
        ], result)

    def testParsePlain(self):
        result = parse_conllu(os.path.join(os.path.dirname(__file__), 'data', u'dataset_plain.conllu'))
        self.assertEqual([
            [
                [u'Ранее', u' ', u'часто', u' ', u'писали', u' ', u'"', u'алгорифм', u'"', u',', u' ', u'сейчас', u' ',
                 u'такое', u' ', u'написание', u' ', u'используется', u' ', u'редко', u',', u' ', u'но', u',', u' ',
                 u'тем', u' ', u'не', u' ', u'менее', u',', u' ', u'имеет', u' ', u'место', u' ', u'(', u'например',
                 u',', u' ', u'Нормальный', u' ', u'алгорифм', u' ', u'Маркова', u')', u'.'],
            ],
            [
                [u'Кто', u' ', u'знает', u',', u' ', u'что', u' ', u'он', u' ', u'там', u' ', u'думал', u'!', u'.',
                 u'.'],
            ],
        ], result)

    def testParseUdpipe(self):
        result = parse_conllu(os.path.join(os.path.dirname(__file__), 'data', u'dataset_udpipe.conllu'))
        self.assertEqual([
            [
                [u'Порву', u'!'],
            ],
            [
                [u'Порву', u'!', u'"', u'...', u'©'],
            ],
            [
                [u'Ребят', u',', u' ', u'я', u' ', u'никому', u' ', u'не', u' ', u'звонила', u'?', u'?', u'?', u'\n',
                 u')))'],
            ],
            [
                [u'Вот', u' ', u'это', u' ', u'был', u' ', u'номер', u'...', u')', u'))', u' ', u'-'],
            ],
        ], result)

    def testParseCollapse(self):
        result = parse_conllu(os.path.join(os.path.dirname(__file__), 'data', u'dataset_collapse.conllu'))
        self.assertEqual([
            [
                [u'Er', u' ', u'arbeitet', u' ', u'fürs', u' ', u'FBI', u' ', u'(', u' ', u'deutsch', u' ', u'etwa',
                 u' ', u':', u' ', u'„', u' ', u'Bundesamt', u' ', u'für', u' ', u'Ermittlung', u' ', u'“', u' ', u')',
                 u' ', u'.'],
            ],
        ], result)


class TestRandomGlue(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def testEmpty(self):
        self.assertEqual([], random_glue())

    def testShape(self):
        result = random_glue(1, 1, 1, 1, 1)
        self.assertEqual([['']], result)

    def testNormal(self):
        result = random_glue(space=10, tab=1, newline=1, reserve=1)
        self.assertEqual([[' '], [' '], [' '], [' '], [' '], [' '], [' '], [' '], ['\t']], result)


class TestAugmentDataset(unittest.TestCase):
    def testEmpty(self):
        result = augment_dataset([])
        self.assertEqual([], result)

    def testNormal(self):
        np.random.seed(5)
        source = [
            [['First', ' ', 'sentence', ' ', 'in', ' ', 'paragraph', '.'],
             ['Second', ' ', '"', 'sentence', '"', ' ', 'in', ' ', 'paragraph', '.']],
            [['Single', ' ', '-', ' ', 'sentence', '.']],
        ]
        expected = [
            [['First', ' ', ' ', 'sentence', ' ', ' ', 'in', ' ', 'paragraph', '.', ' '],
             ['Second', ' ', '"', 'sentence', '"', ' ', 'in', ' ', 'paragraph', '.', ' ']],
            [['Single', ' ', '-', u'\u00A0', 'sentence', '.', ' ']]
        ]
        result = augment_dataset(source)
        self.assertEqual(expected, result)

    def testSpaces(self):
        np.random.seed(2)
        source = [
            [['Single', ' ', 'sentence'],
             ['Next', ' ', 'single', ' ', 'sentence']]
        ]
        expected = [
            [['Single', u' ', 'sentence', '\r\n'],
             ['Next', ' ', 'single', ' ', 'sentence', '\n', '\n']]
        ]
        result = augment_dataset(source)
        self.assertEqual(expected, result)


class TestTokenizeSentence(unittest.TestCase):
    def testEmpty(self):
        result = tokenize_sentence([], [])
        self.assertEqual([], result)

    def testJoinedSide(self):
        result = tokenize_sentence([
            'word', '+', 'word', '-', 'word'
        ], [
            'word+word-word'
        ])
        self.assertEqual(['B', 'N', 'N', 'N', 'N'], result)

    def testJoinedTarget(self):
        result = tokenize_sentence([
            'word+word-word'
        ], [
            'word', '+', 'word', '-', 'word'
        ])
        self.assertEqual(['B'], result)

    def testDifferentJoin(self):
        result = tokenize_sentence([
            'word', '+', 'word-word'
        ], [
            'word+word', '-', 'word'
        ])
        self.assertEqual(['B', 'N', 'N'], result)

    def testNormalJoin(self):
        result = tokenize_sentence([
            'word', '+', 'word-word', '_', 'word'
        ], [
            'word', '+', 'word', '-', 'word_word'
        ])
        self.assertEqual(['B', 'B', 'B', 'N', 'N'], result)


class TestTokenizeDataset(unittest.TestCase):
    def testEmpty(self):
        source = []
        result = tokenize_dataset(source)
        self.assertEqual([], result)

    def testCompex(self):
        source = [
            [[' ', 'test@test.com', ' '],
             [' ', 'www.test.com', ' '],
             [' ', 'word..word', ' '],
             [' ', 'word+word-word', ' '],
             [' ', 'word\\word/word#word', ' ']],
            [[' ', 'test', '@', 'test', '.', 'com', ' '],
             [' ', 'www', '.', 'test', '.', 'com', ' '],
             [' ', 'word', '..', 'word', ' '],
             [' ', 'word', '+', 'word', '-', 'word', ' '],
             [' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' ']],
        ]
        expected = [
            [
                ([' ', 'test', '@', 'test', '.', 'com', ' '],
                 'B,B,N,N,N,N,B'),

                ([' ', 'www', '.', 'test', '.', 'com', ' '],
                 'B,B,N,N,N,N,B'),

                ([' ', 'word', '.', '.', 'word', ' '],
                 'B,B,N,N,N,B'),

                ([' ', 'word', '+', 'word', '-', 'word', ' '],
                 'B,B,N,N,N,N,B'),

                ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                 'B,B,N,N,N,N,N,N,B')
            ],
            [
                ([' ', 'test', '@', 'test', '.', 'com', ' '],
                 'B,B,B,B,B,B,B'),

                ([' ', 'www', '.', 'test', '.', 'com', ' '],
                 'B,B,B,B,B,B,B'),

                ([' ', 'word', '.', '.', 'word', ' '],
                 'B,B,B,N,B,B'),

                ([' ', 'word', '+', 'word', '-', 'word', ' '],
                 'B,B,B,B,B,B,B'),

                ([' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
                 'B,B,B,B,B,B,B,B,B'),
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)

    def testNormal(self):
        source = [
            [['First', ' ', 'sentence', ' ', ' ', ' ', 'in', ' ', 'paragraph', '.', '\n'],
             ['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' ']],
            [['Single', ' ', '-', ' ', 'sentence', '.', '\t']]
        ]
        expected = [
            [
                (['Single', ' ', '-', ' ', 'sentence', '.', '\t'],
                 'B,B,B,B,B,B,B')
            ],
            [
                (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                 'B,B,B,B,B,B,B,B,B'),
                (['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' '],
                 'B,B,B,B,B,B,B,B,B,B,B')
            ]
        ]
        result = tokenize_dataset(source)
        self.assertEqual(expected, result)


class TestMakeDocuments(unittest.TestCase):
    def testEmpty(self):
        source = []
        result = make_documents(source, 2)
        self.assertEqual([], result)

    def testNormal(self):
        source = [
            [
                (['Single', ' ', '-', ' ', 'sentence', '.', '\t'],
                 'B,B,B,B,B,B,B')
            ],
            [
                ([' ', 'test', '@', 'test', '.', 'com', ' '],
                 'B,B,N,N,N,N,B'),
            ],
            [
                (['First', ' ', 'sentence', '   ', 'in', ' ', 'paragraph', '.', '\n'],
                 'B,B,B,B,B,B,B,B,B'),
                (['Second', ' ', '"', 'sentence', '"', ' ', 'in', u'\u00A0', 'paragraph', '.', ' '],
                 'B,B,B,B,B,B,B,B,B,B,B')
            ],
        ]

        expected_documents = [
            u'First sentence   in paragraph.\nSecond "sentence" in\u00A0paragraph. ',
            ' test@test.com Single - sentence.\t',
        ]
        expected_tokens = [
            'B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B',
            'B,B,N,N,N,N,B,B,B,B,B,B,B,B'
        ]
        expected_labels = [
            'N,N,N,N,N,N,N,N,B,N,N,N,N,N,N,N,N,N,N,B',
            'N,N,N,N,N,N,B,N,N,N,N,N,N,B'
        ]

        result = make_documents(source, doc_size=15)
        result_documents, result_tokens, result_labels = zip(*result)
        result_documents, result_tokens, result_labels = \
            list(result_documents), list(result_tokens), list(result_labels)

        self.assertEqual(expected_documents, result_documents)
        self.assertEqual(expected_tokens, result_tokens)
        self.assertEqual(expected_labels, result_labels)


class TestWriteDataset(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = [
            (
                u'First sentence   in paragraph.\nSecond "sentence" in\u00A0paragraph. ',
                'B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B,B',
                'N,N,N,N,N,N,N,N,B,N,N,N,N,N,N,N,N,N,N,B',
            ),
            (
                'Single - sentence.\t test@test.com ',
                'B,B,B,B,B,B,B,B,B,N,N,N,N,B',
                'N,N,N,N,N,N,B,N,N,N,N,N,N,B',
            )
        ]

        write_dataset(self.temp_dir, 'buffer', source)

        wildcard = os.path.join(self.temp_dir, '*.tfrecords.gz')
        dataset = train_input_fn(wildcard, 1, 1, 1)
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        document = features['document']
        tokens = labels['tokens']
        sentences = labels['sentences']

        with self.test_session() as sess:
            document_value, tokens_value, sentences_value = sess.run([document, tokens, sentences])
            self.assertEqual(source[0][0], document_value[0].decode('utf-8'))
            self.assertEqual(source[0][1], b','.join(tokens_value[0]).decode('utf-8'))
            self.assertEqual(source[0][2], b','.join(sentences_value[0]).decode('utf-8'))

            document_value, labels_value = sess.run([document, labels])
            self.assertEqual(source[1][0], document_value[0].decode('utf-8'))
            self.assertEqual(source[0][1], b','.join(tokens_value[0]).decode('utf-8'))
            self.assertEqual(source[0][2], b','.join(sentences_value[0]).decode('utf-8'))
