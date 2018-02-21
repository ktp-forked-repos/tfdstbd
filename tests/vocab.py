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

    def test_fit(self):
        vocab = Vocabulary()
        vocab.fit(['1', ' ', '2', ' ', '1', '\n', '2', '\t', '3', '.'])
        self.assertEqual(vocab.items(), [' ', '1', '2', '\t', '\n', '.', '3'])

    def test_fit_trim(self):
        vocab = Vocabulary()
        vocab.fit(['1', ' ', '2', ' ', '1', '\n', '2', '\t', '3', '.'])
        vocab.trim(2)
        self.assertEqual(vocab.items(), [' ', '1', '2'])

    def test_save_load(self):
        vocab_filename = os.path.join(self.temp_dir, 'vocab.pkl')

        vocab1 = Vocabulary()
        vocab1.fit(['1', ' ', '2', ' ', '1', '\n', '2', '\t', '3', '.'])
        vocab1.save(vocab_filename)

        vocab2 = Vocabulary.load(vocab_filename)
        self.assertEqual(vocab1.items(), vocab2.items())

        vocab1.trim(2)
        self.assertNotEqual(vocab1.items(), vocab2.items())


if __name__ == '__main__':
    unittest.main()
