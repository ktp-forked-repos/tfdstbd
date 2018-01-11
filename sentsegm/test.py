from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

# if __package__ is None:
#     import os, sys
#
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
#     from tokenizer import WordTokenizer, SubWordTokenizer
# else:
from lib.tokenizer import WordTokenizer, SubWordTokenizer


class TestSubWordTokenizer(unittest.TestCase):
    def setUp(self):
        self.word_tokenizer = SubWordTokenizer()

    def test_empty(self):
        tokens = self.word_tokenizer.tokenize('')
        self.assertEqual(tokens, [])

    def test_single(self):
        tokens = self.word_tokenizer.tokenize('Hello')
        self.assertEqual(tokens, ['Hello'])

    def test_quotes(self):
        tokens = self.word_tokenizer.tokenize('"Hello"')
        self.assertEqual(tokens, ['"', 'Hello', '"'])

    def test_brackets(self):
        tokens = self.word_tokenizer.tokenize('[({<word>})]')
        self.assertEqual(tokens, ['[', '(', '{', '<', 'word', '>', '}', ')', ']'])

    def test_punkt(self):
        tokens = self.word_tokenizer.tokenize('word.')
        self.assertEqual(tokens, ['word', '.'])

        tokens = self.word_tokenizer.tokenize('word..')
        self.assertEqual(tokens, ['word', '..'])

        tokens = self.word_tokenizer.tokenize('word...')
        self.assertEqual(tokens, ['word', '...'])

        tokens = self.word_tokenizer.tokenize('word,')
        self.assertEqual(tokens, ['word', ','])

        tokens = self.word_tokenizer.tokenize('word.,')
        self.assertEqual(tokens, ['word', '.', ','])

        tokens = self.word_tokenizer.tokenize('word:')
        self.assertEqual(tokens, ['word', ':'])

        tokens = self.word_tokenizer.tokenize('word;')
        self.assertEqual(tokens, ['word', ';'])

        tokens = self.word_tokenizer.tokenize('word!')
        self.assertEqual(tokens, ['word', '!'])

        tokens = self.word_tokenizer.tokenize('word?')
        self.assertEqual(tokens, ['word', '?'])

    def test_complex(self):
        tokens = self.word_tokenizer.tokenize('test@test.com')
        self.assertEqual(tokens, ['test', '@', 'test', '.', 'com'])

        tokens = self.word_tokenizer.tokenize('www.test.com')
        self.assertEqual(tokens, ['www', '.', 'test', '.', 'com'])

    def test_negative(self):
        tokens = self.word_tokenizer.tokenize('word..word')
        self.assertEqual(tokens, ['word', '..', 'word'])

        tokens = self.word_tokenizer.tokenize('word+word-word')
        self.assertEqual(tokens, ['word', '+', 'word', '-', 'word'])

        tokens = self.word_tokenizer.tokenize('word\\word/word#word')
        self.assertEqual(tokens, ['word', '\\', 'word', '/', 'word', '#', 'word'])


class TestWordTokenizer(unittest.TestCase):
    def setUp(self):
        self.word_tokenizer = WordTokenizer()

    def test_empty(self):
        tokens = self.word_tokenizer.tokenize('')
        self.assertEqual(tokens, [])

    def test_single(self):
        tokens = self.word_tokenizer.tokenize('Hello')
        self.assertEqual(tokens, ['Hello'])

    def test_spacelike(self):
        tokens = self.word_tokenizer.tokenize('Hel\n\n\tlo')
        self.assertEqual(tokens, ['Hel', '\n', '\n', '\t', 'lo'])

        tokens = self.word_tokenizer.tokenize(u'Hel\u00A0lo')
        self.assertEqual(tokens, ['Hel', ' ', 'lo'])

        tokens = self.word_tokenizer.tokenize(u'Hel\r\v\u00A0lo')
        self.assertEqual(tokens, ['Hel', ' ', 'lo'])

    def test_complex(self):
        tokens = self.word_tokenizer.tokenize('Hello,  "My\t\n\n1D-dear!", friend`s.')
        self.assertEqual(tokens, [
            'Hello', ',', ' ', ' ', '"', 'My', '\t', '\n', '\n', '1D', '-', 'dear', '!', '"', ',', ' ', 'friend`s', '.'
        ])


if __name__ == '__main__':
    unittest.main()
