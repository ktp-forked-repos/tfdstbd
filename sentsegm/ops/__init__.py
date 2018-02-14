from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
from tensorflow.python.framework import ops
import tensorflow as tf

_ops_module = tf.load_op_library(path.join(path.dirname(path.abspath(__file__)), 'ops.so'))


def split_tokens(source):
    """Split source strings into tokens almost without transformation (except Unicode NFC normalization).
    Result tokens could be easy joined to obtain source strings.

    Args:
        source: `1-D` string `Tensor`, strings to split
    Raises:
        ValueError: If source strings are not UTF-8 sequences.
    Returns:
        A `SparseTensor` of rank `2`, the tokens from source strings.
    """

    source = tf.convert_to_tensor(source, dtype=tf.string)
    indices, values, shape = _ops_module.split_tokens(source)
    indices.set_shape([None, 2])
    values.set_shape([None])
    shape.set_shape([2])

    return tf.SparseTensor(indices, values, shape)

ops.NotDifferentiable("SplitTokens")


def extract_features(source):
    """Extract features from source string tokens.

    Args:
        source: `2-D` string `SparseTensor`, strings to extract features
    Raises:
        ValueError: If source strings are not UTF-8 sequences.
    Returns:
        A `SparseTensor` of rank `3`, the features from tokens.
    """

    indices, values, shape = _ops_module.extract_features(source.indices, source.values, source.dense_shape)
    indices.set_shape([None, 3])
    values.set_shape([None])
    shape.set_shape([3])

    return tf.SparseTensor(indices, values, shape)

ops.NotDifferentiable("ExtractFeatures")
