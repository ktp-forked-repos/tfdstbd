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


# def serve_input_fn():
#     example_proto = tf.placeholder(dtype=tf.string, shape=[1], name='input_example')
#     receiver_tensors = {'examples': example_proto}
#
#     parsed_features = tf.parse_example(
#         example_proto,
#         features={
#             'document': tf.FixedLenFeature(1, tf.string)
#         })
#
#     return tf.estimator.export.ServingInputReceiver(parsed_features, receiver_tensors)

def serve_input_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='examples')
    features = {
        'documents': serialized_tf_example
    }
    # features = tf.parse_example(serialized_tf_example, {
    #     'word': tf.FixedLenFeature(1, tf.string)
    # })
    # features['word'] = features['word'][0]

    return tf.estimator.export.ServingInputReceiver(features, serialized_tf_example)
