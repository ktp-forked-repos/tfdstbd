# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import unittest
from sentsegm.vocab import Vocabulary


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testFit(self):
        vocab = Vocabulary()
        vocab.fit(['1', ' ', '2', ' ', '1', '\n', '2', '\t', '3', '.'])
        self.assertEqual(vocab.items(), [' ', '1', '2', '\t', '\n', '.', '3'])

    def testFitTrim(self):
        vocab = Vocabulary()
        vocab.fit(['1', ' ', '2', ' ', '1', '\n', '2', '\t', '3', '.'])
        vocab.trim(2)
        self.assertEqual(vocab.items(), [' ', '1', '2'])

    def testSaveLoad(self):
        vocab_filename = os.path.join(self.temp_dir, 'vocab.pkl')

        vocab1 = Vocabulary()
        vocab1.fit(['1', ' ', '2', ' ', '1', '\n', '2', '\t', '3', '.'])
        vocab1.save(vocab_filename)

        vocab2 = Vocabulary.load(vocab_filename)
        self.assertEqual(vocab1.items(), vocab2.items())

        vocab1.trim(2)
        self.assertNotEqual(vocab1.items(), vocab2.items())

    def testSaveTsv(self):
        vocab_filename = os.path.join(self.temp_dir, 'vocab.tsv')

        vocab = Vocabulary()
        vocab.fit(['1', ' ', u'2', ' ', '1', '\n', '2', '\t', u'а', '.'])
        vocab.save(vocab_filename, False)
        expected = u'token\tfrequency\nSPACE_32\t2\n1\t2\n2\t2\nSPACE_9\t1\nSPACE_10\t1\n.\t1\nа\t1\n'

        with open(vocab_filename, 'rb') as vf:
            result = vf.read().decode('utf-8')
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
