from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfucops import transform_lower_case, transform_upper_case, expand_split_chars, transform_normalize_unicode, \
    transform_zero_digits, transform_wrap_with, expand_char_ngrams
from . import features_length_case


def extract_features(tokens):
    source = tf.sparse_tensor_to_dense(tokens, default_value='')

    return features_length_case(source)


def extract_ngrams(tokens, minn, maxn):
    # Normalize unicode tokens with NFD algorithm
    ugly = transform_normalize_unicode(tokens, 'NFKC')

    # Make tokens case lower
    ugly = transform_lower_case(ugly)

    # Replace each digit with 0
    ugly = transform_zero_digits(ugly)

    # Add start & end character to each token
    ugly = transform_wrap_with(ugly, '<', '>')

    # Extract character ngrams with word itself
    ngrams = expand_char_ngrams(ugly, minn, maxn, itself='ALWAYS')

    return ngrams
