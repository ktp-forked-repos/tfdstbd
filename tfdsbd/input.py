from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def train_input_fn(wildcard, batch_size):
    with tf.name_scope('input'):
        # Create dataset from multiple TFRecords files
        files = tf.data.Dataset.list_files(wildcard)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=3)

        # Parse serialized examples
        def _parse_example(example_proto):
            example = tf.parse_single_example(
                example_proto,
                features={
                    'document': tf.FixedLenFeature(1, tf.string),
                    'labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                })

            return {'documents': example['document'][0]}, example['labels']

        dataset = dataset.map(_parse_example)

        # Create padded batch
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=({'documents': []}, [None]),
            padding_values=({'documents': ''}, tf.cast(0, dtype=tf.int64))
        )
        dataset = dataset.prefetch(10)

        return dataset


def predict_input_fn(document):
    # Create dataset from document list
    dataset = tf.data.Dataset.from_tensors({'documents': document})

    # Create batch
    dataset = dataset.batch(1)

    return dataset
