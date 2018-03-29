from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfucops import transform_lower_case, transform_upper_case, expand_split_chars, transform_normalize_unicode, transform_zero_digits, transform_wrap_with, expand_char_ngrams


def extract_features(tokens_source):
    tokens_lower = transform_lower_case(tokens_source)
    tokens_upper = transform_upper_case(tokens_source)

    feat_no_case = tf.equal(tokens_lower, tokens_upper)

    has_case = tf.logical_not(feat_no_case)
    feat_lower_case = tf.logical_and(has_case, tf.equal(tokens_lower, tokens_source))
    feat_upper_case = tf.logical_and(has_case, tf.equal(tokens_upper, tokens_source))

    chars_source = expand_split_chars(tokens_source, default='')

    chars_shape_size = tf.size(tf.shape(chars_source))

    first_chars_slice_begin = tf.zeros([chars_shape_size], dtype=tf.int32)
    first_chars_slice_size = tf.concat([
        tf.fill([chars_shape_size - 1], -1),
        [1]
    ], axis=-1)
    first_chars = tf.slice(chars_source, first_chars_slice_begin, first_chars_slice_size)
    first_chars_upper = transform_upper_case(first_chars)

    last_chars_slice_begin = tf.concat([
        tf.zeros([chars_shape_size - 1], dtype=tf.int32),
        [1]
    ], axis=-1)
    last_chars_slice_size = tf.fill([chars_shape_size], -1)
    last_chars = tf.slice(chars_source, last_chars_slice_begin, last_chars_slice_size)
    last_chars_lower = transform_lower_case(last_chars)

    feat_title_case = tf.logical_and(
        has_case,
        tf.logical_and(
            tf.squeeze(tf.equal(first_chars_upper, first_chars), axis=-1),
            tf.cast(
                tf.reduce_min(
                    tf.cast(
                        tf.equal(last_chars_lower, last_chars),
                        tf.int32
                    ),
                    axis=-1
                ),
                dtype=tf.bool
            )
        )
    )

    chars_mask = tf.not_equal(chars_source, '')
    chars_length = tf.reduce_sum(tf.cast(chars_mask, dtype=tf.int32), axis=-1)
    feat_smooth_length = tf.sigmoid(tf.cast(chars_length, tf.float32) - 5.)

    all_features = [
        tf.cast(feat_no_case, tf.float32),
        tf.cast(feat_lower_case, tf.float32),
        tf.cast(feat_upper_case, tf.float32),
        tf.cast(feat_title_case, tf.float32),
        feat_smooth_length
    ]

    return tf.stack(all_features, axis=-1)


def extract_ngrams(tokens_source, minn, maxn):
    # Normalize unicode tokens with NFD algorithm
    tokens = transform_normalize_unicode(tokens_source, 'NFKC')

    # Make tokens case lower
    tokens = transform_lower_case(tokens)

    # Replace each digit with 0
    tokens = transform_zero_digits(tokens)

    # Add start & end character to each token
    tokens = transform_wrap_with(tokens, '<', '>')

    # Extract character ngrams with word itself
    ngrams = expand_char_ngrams(tokens, minn, maxn, itself='ALWAYS')

    return ngrams