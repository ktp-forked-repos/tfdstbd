from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfucops import transform_normalize_unicode, expand_split_words, transform_lower_case, transform_zero_digits, \
    transform_wrap_with, expand_char_ngrams
from .transform import extract_features, extract_ngrams
from .metric import f1_score


def sbd_model_fn(features, labels, mode, params):
    with tf.device('/cpu:0'):
        docs = features['documents']

        # Add tokinization layer
        with tf.name_scope('tokens'):
            # Normalize unicode documents with NFC algorithm
            docs = transform_normalize_unicode(docs, 'NFC')
            max_docs = tf.size(docs)

            # Split documents to tokens
            tokens = expand_split_words(docs)

            # Compute padded tokens mask
            tokens_masks = tf.SparseTensor(
                indices=tokens.indices,
                values=tf.ones_like(tokens.values, dtype=tf.int32),
                dense_shape=tokens.dense_shape
            )
            tokens_masks = tf.sparse_tensor_to_dense(tokens_masks, default_value=0)

            # Compute actual documents shape (in term of tokens)
            tokens_lengths = tf.reduce_sum(tokens_masks, axis=-1)
            max_tokens = tf.reduce_max(tokens_lengths)

        # Add ngrams extraction layer
        with tf.name_scope('ngrams'):
            # Split tokens to ngrams
            ngrams = extract_ngrams(tokens, params.min_n, params.max_n)

            # Compute padded ngrams mask
            ngrams_masks = tf.SparseTensor(
                indices=ngrams.indices,
                values=tf.ones_like(ngrams.values, dtype=tf.int32),
                dense_shape=ngrams.dense_shape
            )

            # Compute actual documents shape (in term of ngrams)
            ngrams_lengths = tf.sparse_reduce_sum_sparse(ngrams_masks, axis=-1)
            max_ngrams = tf.sparse_reduce_max(ngrams_lengths)

        # Add ngrams embedding layer
        with tf.name_scope('embedings'):
            # Reshape ngrams to flat list, required to use feature columns
            ngrams_flat = tf.sparse_reshape(ngrams, [-1, max_ngrams])
            ngrams_flat, _ = tf.sparse_fill_empty_rows(ngrams_flat, '')

            vocab_table = tf.contrib.lookup.index_table_from_tensor(
                params.ngram_vocab,
                num_oov_buckets=params.uniq_count
            )
            ngrams_embeddings = tf.get_variable(
                'ngrams_embeddings',
                [len(params.ngram_vocab) + params.uniq_count, params.embed_size],
                dtype=None,
                initializer=tf.random_uniform_initializer(-1, 1),
            )
            # ngrams_embeddings = tf.nn.dropout(
            #     ngrams_embeddings,
            #     keep_prob=params.keep_prob,
            #     noise_shape=[len(params.ngram_vocab) + params.uniq_count, 1]
            # )

            ngrams_ids = tf.SparseTensor(
                indices=ngrams_flat.indices,
                values=vocab_table.lookup(ngrams_flat.values),
                dense_shape=ngrams_flat.dense_shape
            )
            embeddings = tf.nn.embedding_lookup_sparse(
                ngrams_embeddings,
                ngrams_ids,
                None,
                combiner='mean'
            )
            embeddings = tf.reshape(embeddings, [max_docs, max_tokens, params.embed_size])

        # Add feature extraction layer
        with tf.name_scope('features'):
            features = extract_features(tokens)

        # Add embedding+features layer
        with tf.name_scope('concat'):
            features_size = 6
            inputs = tf.concat([embeddings, features], axis=-1)
            # inputs = embeddings

    # Add recurrent layer
    with tf.name_scope('rnn'):
        rnn_directions = 2
        if params.use_cudnn:
            # CudnnLSTM is time-major
            inputs = tf.transpose(inputs, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=params.rnn_layers,
                num_units=params.rnn_size,
                dropout=1 - params.keep_prob if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
                direction='bidirectional'
            )
            rnn_output, _ = lstm(inputs)
            # Convert back from time-major outputs to batch-major outputs
            rnn_output = tf.transpose(rnn_output, [1, 0, 2])
        else:
            cells_fw = [tf.contrib.rnn.LSTMBlockCell(params.rnn_size) for _ in range(params.rnn_layers)]
            cells_bw = [tf.contrib.rnn.LSTMBlockCell(params.rnn_size) for _ in range(params.rnn_layers)]
            if mode == tf.estimator.ModeKeys.TRAIN and params.keep_prob < 1:
                cells_fw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params.keep_prob) for cell in cells_fw]
                cells_bw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params.keep_prob) for cell in cells_bw]
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
            # 'tokens': tokens,
            'tokens': tf.sparse_tensor_to_dense(tokens, default_value=''),
            'classes': tf.argmax(logits, axis=2),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
        )

    # Add the loss
    with tf.name_scope('loss'):
        # loss = tf.losses.sigmoid_cross_entropy(
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
    with tf.name_scope('train'):
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
