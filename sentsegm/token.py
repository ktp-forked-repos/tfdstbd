from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import re
from nltk.tokenize.treebank import TreebankWordTokenizer

try:
    # Python 2
    from future_builtins import filter, map
except ImportError:
    # Python 3
    pass


class SubWordTokenizer(TreebankWordTokenizer):
    # starting quotes
    STARTING_QUOTES = [
        # Simplified with str.replace
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),

        # Disabled...
        # (re.compile(r'\.\.\.'), r' ... '),
        # ... due to more complex expression
        (re.compile(r'(\.+)'), r' \1 '),

        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),  # Handles the final period.
        (re.compile(r'[?!]'), r' \g<0> '),

        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        # Removed, not used
    ]

    # ending quotes
    ENDING_QUOTES = [
        # Simplified with str.replace
    ]

    def tokenize(self, text, *args):
        if text.isspace() or text.isalnum():
            return [text]

        text = text.replace('"', ' " ').replace('-', ' - ').replace('+', ' + ')
        text = text.replace('\\', ' \\ ').replace('/', ' / ')

        return super(SubWordTokenizer, self).tokenize(text)


class WordTokenizer():
    def __init__(self):
        self.sub_word_tokenizer = SubWordTokenizer()

    @staticmethod
    def _split_item_by(item, separator):
        glue = ' ' if separator is None else separator
        tokens = [item] if item.isspace() else item.split(separator)
        tokens = itertools.chain.from_iterable([glue, t] for t in tokens)
        tokens = itertools.islice(tokens, 1, None)
        tokens = filter(len, tokens)

        return tokens

    @staticmethod
    def _split_list_by(items, separator):
        tokens = map(WordTokenizer._split_item_by, items, itertools.repeat(separator))
        tokens = itertools.chain.from_iterable(tokens)

        return tokens

    def tokenize(self, text):
        split_by_space = WordTokenizer._split_item_by(text, ' ')
        split_by_tab = WordTokenizer._split_list_by(split_by_space, '\t')
        split_by_newline = WordTokenizer._split_list_by(split_by_tab, '\n')
        split_by_spacelike = WordTokenizer._split_list_by(split_by_newline, None)

        final_split = map(self.sub_word_tokenizer.tokenize, split_by_spacelike)
        final_split = itertools.chain.from_iterable(final_split)

        return list(final_split)
