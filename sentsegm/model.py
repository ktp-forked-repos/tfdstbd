from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .metric import f1_score


class Model:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    NUM_CLASSES = 2

    def __init__(self, vocab_words, embed_size, rnn_layers, rnn_size, keep_prob):
        vocab_words = [Model.PAD_TOKEN, Model.UNK_TOKEN] + vocab_words
        self.vocab_size = len(vocab_words)
        # vocab_words = tf.constant(vocab_words)
        self.vocab_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=vocab_words,
            num_oov_buckets=0,
            default_value=1  # UNK_TOKEN
        )

        self.embed_size = embed_size
        self.param_rnn_layers = rnn_layers
        self.param_rnn_size = rnn_size
        self.param_keep_prob = keep_prob

        # self.token_embeddings = tf.get_variable('token_embeddings', [vocab_size, embed_size])
        # self.softmax_weight = tf.get_variable('softmax_weight', [2 * self.param_rnn_size, Model.NUM_CLASSES])
        # self.softmax_bias = tf.get_variable('softmax_bias', [Model.NUM_CLASSES])

    def _rnn_cell(self, training):
        cells = []
        for _ in range(self.param_rnn_layers):
            cell = tf.contrib.rnn.GRUCell(self.param_rnn_size)
            if training:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell,
                    input_keep_prob=self.param_keep_prob,
                    output_keep_prob=self.param_keep_prob
                )
            cells.append(cell)

        return tf.contrib.rnn.MultiRNNCell(cells)

    def __call__(self, features, training):
        """Add operations to classify a batch of tokens.
        Args:
          features: A dict with 'document' and 'length' keys
          training: A boolean. Set to True to add operations required only when training the classifier.
        Returns:
          A logits Tensor with shape [<batch_size>, 2].
        """

        word_ids = self.vocab_table.lookup(features['document'])  # tokens -> ids
        # word_ids = tf.feature_column.input_layer(features, self.feature_columns)
        word_vectors = tf.contrib.layers.embed_sequence(
            word_ids,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_size
        )

        # (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        #     self._rnn_cell(training),
        #     self._rnn_cell(training),
        #     word_vectors,
        #     sequence_length=features['length'],
        #     dtype=tf.float32
        # )
        # rnn_output = tf.concat([output_fw, output_bw], 0)
        # rnn_output = tf.reshape(rnn_output, [-1, self.param_rnn_size * 2])

        rnn_output, _ = tf.nn.dynamic_rnn(
            self._rnn_cell(training),
            word_vectors,
            dtype=tf.float32,
            sequence_length=features['length'],
        )
        rnn_output = tf.reshape(rnn_output, [-1, self.param_rnn_size])

        logits = tf.layers.dense(rnn_output, Model.NUM_CLASSES, activation=None)
        logits = tf.reshape(logits, [-1, tf.reduce_max(features['length']), Model.NUM_CLASSES])

        return logits


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = Model(params['vocab_words'], params['embed_size'], params['rnn_layers'], params['rnn_size'],
                  params['keep_prob'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        predictions = {
            'document': features['document'],
            'class': tf.argmax(logits, axis=2),
            'probability': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            # export_outputs={
            #     'classify': tf.estimator.export.PredictOutput(predictions)
            # }
        )

    if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        logits = model(features, training=is_training)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metrics = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(logits, axis=2),
            ),
            'f1_score': f1_score(
                labels=labels,
                predictions=tf.argmax(logits, axis=2),
            )
        }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metrics
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        tf.summary.scalar('accuracy', eval_metrics['accuracy'][1])
        tf.summary.scalar('f1_score', eval_metrics['f1_score'][1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train
        )
