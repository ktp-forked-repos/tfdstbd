from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfucops import expand_split_words, transform_normalize_unicode

PAD_TOKEN = 'PAD'


def train_input_fn(wildcard, batch_size):
    """The input_fn argument for training with Estimator."""

    # Create dataset from multiple TFRecords files
    files = tf.data.TFRecordDataset.list_files(wildcard)
    dataset = files.interleave(
        lambda file: tf.data.TFRecordDataset(file, compression_type='GZIP'),
        cycle_length=5
    )

    # Parse serialized examples
    def _parse_example(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'document': tf.FixedLenFeature(1, tf.string),
                'labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            })

        return {'document': features['document'][0]}, features['labels']

    dataset = dataset.map(_parse_example)

    # Extract features
    def _parse_features(features, labels):
        features['tokens'] = expand_split_words(
            features['document'],
            default_value=PAD_TOKEN
        )
        features['length'] = tf.size(features['tokens'])

        return features, labels

    dataset = dataset.map(_parse_features, num_parallel_calls=10)

    # Create padded batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=({'document': [], 'tokens': [None], 'length': []}, [None]),
        padding_values=({'document': '', 'tokens': PAD_TOKEN, 'length': 0}, tf.cast(0, dtype=tf.int64))
    )
    dataset = dataset.prefetch(10)

    return dataset


def predict_input_fn(document):
    """The input_fn argument for predicting with Estimator."""

    # Create dataset from document list
    dataset = tf.data.Dataset.from_tensors({'document': document})

    # Extract features
    def _parse_features(features):
        features['document'] = transform_normalize_unicode(features['document'], 'NFC')
        features['tokens'] = expand_split_words(
            features['document'],
            default_value=PAD_TOKEN
        )
        features['length'] = tf.size(features['tokens'])

        return features

    dataset = dataset.map(_parse_features)

    # Create batch
    dataset = dataset.batch(1)

    return dataset
