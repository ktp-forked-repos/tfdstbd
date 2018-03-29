from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfucops import transform_normalize_unicode, expand_split_words, transform_lower_case, transform_zero_digits, \
    transform_wrap_with, expand_char_ngrams
from .transform import extract_features, extract_ngrams
from .metric import f1_score


def sbd_model_fn(features, labels, mode, params):
    docs = features['documents']

    # Add tokinization layer
    with tf.name_scope('tokens'):
        # Normalize unicode documents with NFC algorithm
        docs = transform_normalize_unicode(docs, 'NFC')
        max_docs = tf.size(docs)

        # Split documents to tokens
        tokens = expand_split_words(docs, default='')

        # Compute padded tokens mask
        tokens_masks = tf.cast(tf.not_equal(tokens, ''), dtype=tf.int32)

        # Compute actual documents shape (in term of tokens)
        tokens_lengths = tf.reduce_sum(tokens_masks, axis=-1)
        max_tokens = tf.reduce_max(tokens_lengths)

    # Add ngrams extraction layer
    with tf.name_scope('ngrams'):
        # Split tokens to ngrams
        ngrams = extract_ngrams(tokens, params.min_n, params.max_n)

        # Compute padded ngrams mask
        ngrams_masks = tf.cast(tf.not_equal(ngrams, ''), dtype=tf.int32)

        # Compute actual documents shape (in term of ngrams)
        ngrams_lengths = tf.reduce_sum(ngrams_masks, axis=-1)
        max_ngrams = tf.reduce_max(ngrams_lengths)

        # Reshape ngrams to flat list, required to use feature columns
        ngrams = tf.reshape(ngrams, [-1])

    # Add ngrams embedding layer
    with tf.name_scope('embedings'):

        # Pass ngrams through features input layer
        # hashed_column = tf.feature_column.categorical_column_with_hash_bucket(
        #     key='ngrams',
        #     hash_bucket_size=1000000
        # )
        vocab_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='ngrams',
            vocabulary_list=params.ngram_vocab,
            num_oov_buckets=1000
        )
        embedding_column = tf.feature_column.embedding_column(
            categorical_column=vocab_column,
            dimension=params.embed_size
        )
        embeddings = tf.feature_column.input_layer({
            'ngrams': ngrams
        }, [embedding_column])

        # Reshape embeddings to original ngrmas shape
        embeddings = tf.reshape(embeddings, [max_docs, max_tokens, max_ngrams, params.embed_size])

        # Mask padded ngrams
        embeddings = tf.multiply(embeddings, tf.expand_dims(tf.cast(ngrams_masks, tf.float32), -1))

        # Compute mean for ngrams to get token embeddings
        embeddings = tf.reduce_sum(embeddings, axis=-2)
        embeddings = tf.divide(embeddings, tf.expand_dims(tf.cast(ngrams_lengths, dtype=tf.float32), -1))

    # Add feature extraction layer
    with tf.name_scope('features'):
        features = extract_features(tokens)

    # Add embedding+features layer
    with tf.name_scope('concat'):
        features_size = 4
        inputs = tf.concat([embeddings, features], axis=-1)

    # Add recurrent layer
    with tf.name_scope('rnn'):
        cells_fw = [tf.contrib.rnn.GRUCell(params.rnn_size) for _ in range(params.rnn_layers)]
        cells_bw = [tf.contrib.rnn.GRUCell(params.rnn_size) for _ in range(params.rnn_layers)]
        if mode == tf.estimator.ModeKeys.TRAIN and params.keep_prob < 1:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params.keep_prob) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params.keep_prob) for cell in cells_bw]
        rnn_directions = 2
        rnn_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=inputs,
            sequence_length=tokens_lengths,
            dtype=tf.float32
        )

    # Add fully-connected layer
    with tf.name_scope('dense'):
        # Flatten to apply same weights to all time steps
        flat = tf.reshape(rnn_output, [
            -1,
            params.rnn_size * rnn_directions
        ])
        num_classes = 2
        logits = tf.layers.dense(
            inputs=flat,
            units=num_classes,
            activation=None  # or with activation?
        )
        logits = tf.reshape(logits, [
            max_docs,
            max_tokens,
            num_classes
        ])

    # Build EstimatorSpec's
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'tokens': tokens,
            'classes': tf.argmax(logits, axis=2),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
        )

    # Add the loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        weights=tokens_masks
    )

    # Add metrics
    with tf.name_scope('metrics'):
        accuracy_metric = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.argmax(logits, axis=2),
        )
        f1_metric = f1_score(
            labels=labels,
            predictions=tf.argmax(logits, axis=2),
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': accuracy_metric,
                'f1': f1_metric
            }
        )

    # Add the optimizer
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer='Adam',
    )

    assert mode == tf.estimator.ModeKeys.TRAIN
    metrics_hook = tf.train.LoggingTensorHook({
        'accuracy': accuracy_metric[1],
        'f1': f1_metric[1]
    }, every_n_iter=100)

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
        training_hooks=[metrics_hook]
    )
