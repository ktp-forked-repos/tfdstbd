from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfscc3d import sequence_categorical_column_with_vocabulary_list
from tensorflow import feature_column as core_columns
from tensorflow.contrib import feature_column as contrib_columns
from tfunicode import expand_split_words
from .feature import extract_case_length_features, extract_ngram_features


def input_feature_columns(ngram_vocab, ngram_dimension, ngram_oov=1, ngram_combiner='sum'):
    ngram_categorial_column = sequence_categorical_column_with_vocabulary_list(
        key='ngrams',
        vocabulary_list=ngram_vocab,
        dtype=tf.string,
        num_oov_buckets=ngram_oov,
    )
    ngram_embedding_column = core_columns.embedding_column(
        categorical_column=ngram_categorial_column,
        dimension=ngram_dimension,
        combiner=ngram_combiner,
    )

    return [
        ngram_embedding_column,
        contrib_columns.sequence_numeric_column('length'),
        contrib_columns.sequence_numeric_column('no_case'),
        contrib_columns.sequence_numeric_column('lower_case'),
        contrib_columns.sequence_numeric_column('upper_case'),
        contrib_columns.sequence_numeric_column('title_case'),
        contrib_columns.sequence_numeric_column('mixed_case'),
    ]


def train_input_fn(wild_card, batch_size, ngram_minn, ngram_maxn):
    with tf.name_scope('input'):
        # Create dataset from multiple TFRecords files
        files = tf.data.Dataset.list_files(wild_card)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=5)

        dataset = dataset.batch(batch_size)

        def _parse_examples(examples_proto):
            examples = tf.parse_example(
                examples_proto,
                features={
                    'document': tf.FixedLenFeature(1, tf.string),
                    'labels': tf.VarLenFeature(tf.string),
                })

            documents = tf.squeeze(examples['document'], axis=1)
            words = expand_split_words(documents)
            length, no_case, lower_case, upper_case, title_case, mixed_case = extract_case_length_features(words)
            ngrams = extract_ngram_features(words, ngram_minn, ngram_maxn)

            labels = tf.sparse_tensor_to_dense(examples['labels'], default_value='N')

            return {
                       'documents': documents,
                       'words': words,
                       'words_out': tf.sparse_tensor_to_dense(words, default_value=''),
                       'ngrams': ngrams,
                       'length': length,
                       'no_case': no_case,
                       'lower_case': lower_case,
                       'upper_case': upper_case,
                       'title_case': title_case,
                       'mixed_case': mixed_case,
                   }, labels

        dataset = dataset.map(_parse_examples, num_parallel_calls=32)

        dataset = dataset.prefetch(10)

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

# def serve_input_fn():
#     serialized_tf_example = tf.placeholder(dtype=tf.string,
#                                            shape=[None],
#                                            name='examples')
#     features = {
#         'documents': serialized_tf_example
#     }
#     # features = tf.parse_example(serialized_tf_example, {
#     #     'word': tf.FixedLenFeature(1, tf.string)
#     # })
#     # features['word'] = features['word'][0]
#
#     return tf.estimator.export.ServingInputReceiver(features, serialized_tf_example)
