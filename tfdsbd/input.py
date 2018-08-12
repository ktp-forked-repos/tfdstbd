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
        contrib_columns.sequence_numeric_column('word_length'),
        contrib_columns.sequence_numeric_column('is_no_case'),
        contrib_columns.sequence_numeric_column('is_lower_case'),
        contrib_columns.sequence_numeric_column('is_upper_case'),
        contrib_columns.sequence_numeric_column('is_title_case'),
        contrib_columns.sequence_numeric_column('is_mixed_case'),
    ]


def features_from_documens(documents, ngram_minn, ngram_maxn):
    words = expand_split_words(documents)  # Transformation should be equal with train dataset tokenization
    length, no_case, lower_case, upper_case, title_case, mixed_case = extract_case_length_features(words)
    ngrams = extract_ngram_features(words, ngram_minn, ngram_maxn)

    return {
        'documents': documents,
        'words': tf.sparse_tensor_to_dense(words, default_value=''),  # Required to pass in prediction
        'ngrams': ngrams,
        'word_length': length,
        'is_no_case': no_case,
        'is_lower_case': lower_case,
        'is_upper_case': upper_case,
        'is_title_case': title_case,
        'is_mixed_case': mixed_case,
    }


def train_input_fn(wild_card, batch_size, ngram_minn, ngram_maxn):
    with tf.name_scope('input'):
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
            labels = tf.sparse_tensor_to_dense(examples['labels'], default_value='B')
            features = features_from_documens(documents, ngram_minn, ngram_maxn)

            return features, labels

        dataset = dataset.map(_parse_examples, num_parallel_calls=32)
        dataset = dataset.prefetch(10)

        return dataset


def serve_input_fn(ngram_minn, ngram_maxn):
    def serving_input_receiver_fn():
        documents = tf.placeholder(dtype=tf.string, shape=[None], name='documents')
        features = features_from_documens(documents, ngram_minn, ngram_maxn)

        return tf.estimator.export.ServingInputReceiver(features, documents)

    return serving_input_receiver_fn
