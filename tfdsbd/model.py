from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .metric import f1_score


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""

    def _get_input_tensors(features, labels):
        """Converts the input dict into tokens, lengths, and labels tensors."""
        return features['tokens'], features['length'], labels

    def _add_embed_layers(tokens):
        """Adds embedding layers."""
        # words = tf.constant(params.vocab_words)
        vocab = tf.contrib.lookup.index_table_from_tensor(
            mapping=params.vocab_words,
            num_oov_buckets=1,  # unique tokens all-in-one
        )
        ids = vocab.lookup(tokens)  # tokens -> ids
        embeddings = tf.contrib.layers.embed_sequence(
            ids=ids,
            vocab_size=len(params.vocab_words) + 1,  # or size + 1 ?
            # vocab_size=vocab.size(),  # or size + 1 ?
            embed_dim=params.embed_size,
        )
        return embeddings

    def _add_rnn_layers(embedded, lengths):
        """Adds recurrent neural network layers."""
        cells_fw = [tf.contrib.rnn.GRUCell(params.rnn_size) for _ in range(params.rnn_layers)]
        cells_bw = [tf.contrib.rnn.GRUCell(params.rnn_size) for _ in range(params.rnn_layers)]
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     cells_fw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params.keep_prob) for cell in cells_fw]
        #     cells_bw = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params.keep_prob) for cell in cells_bw]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=embedded,
            sequence_length=lengths,
            dtype=tf.float32
        )
        return outputs

    def _add_fc_layers(final_state):
        """Adds a fully connected layer."""
        batch_size = tf.shape(final_state)[0]
        max_length = tf.shape(final_state)[1]

        # Flatten to apply same weights to all time steps
        flat = tf.reshape(final_state, [
            -1,
            params.rnn_size * 2
        ])
        logits = tf.layers.dense(
            inputs=flat,
            units=2,  # nuber of classes
            activation=None  # or with activation?
        )
        logits = tf.reshape(logits, [
            batch_size,
            max_length,
            2  # number of classes
        ])
        return logits

    # Build the model.
    tokens, lengths, labels = _get_input_tensors(features, labels)
    embeddings = _add_embed_layers(tokens)
    final_state = _add_rnn_layers(embeddings, lengths)
    logits = _add_fc_layers(final_state)

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
            # export_outputs={
            #     'classify': tf.estimator.export.PredictOutput(predictions)
            # }
        )

    # Add the loss.
    mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
    mask = tf.cast(mask, dtype=tf.int32)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        weights=mask  # 0 for padded tokens, should reduce padded examples loss to 0
    )

    # Add metrics
    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=tf.argmax(logits, axis=2),
    )
    f1 = f1_score(
        labels=labels,
        predictions=tf.argmax(logits, axis=2),
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': accuracy,
                'f1': f1
            }
        )

    # Add the optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer='Adam',
        # summaries=['learning_rate', 'loss']
    )
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('f1_score', f1[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op
        )
