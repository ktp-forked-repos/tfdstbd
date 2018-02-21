from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .ops import split_tokens
from .metrics import f1_score


class Model:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    NUM_CLASSES = 2

    def __init__(self, vocab_words, embed_size, rnn_layers, rnn_size, keep_prob):
        # PAD_TOKEN should have index==0 to calculate real sequence length later in padded batch
        vocab_words = [Model.PAD_TOKEN, Model.UNK_TOKEN] + vocab_words
        vocab_size = len(vocab_words)
        # vocab_words = tf.constant(vocab_words)
        self.vocab_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=vocab_words,
            num_oov_buckets=0,
            default_value=1  # UNK_TOKEN
        )

        self.param_rnn_layers = rnn_layers
        self.param_rnn_size = rnn_size
        self.param_keep_prob = keep_prob

        self.token_embeddings = tf.get_variable('token_embeddings', [vocab_size, embed_size])
        self.softmax_weight = tf.get_variable('softmax_weight', [2 * self.param_rnn_size, Model.NUM_CLASSES])
        self.softmax_bias = tf.get_variable('softmax_bias', [Model.NUM_CLASSES])

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

    def __call__(self, documents, training):
        """Add operations to classify a batch of tokens.
        Args:
          inputs: A Tensor representing a batch of tokenized documents.
          training: A boolean. Set to True to add operations required only when training the classifier.
        Returns:
          A logits Tensor with shape [<batch_size>, 2].
        """

        document_tokens = split_tokens(documents)  # documents -> tokens
        document_tokens = tf.sparse_tensor_to_dense(document_tokens, default_value=Model.PAD_TOKEN)

        token_ids = self.vocab_table.lookup(document_tokens)  # tokens -> ids
        inputs_mask = tf.sign(token_ids)  # 0 for padded tokens, 1 for real ones
        inputs_length = tf.cast(tf.reduce_sum(inputs_mask, 1), tf.int32)  # real tokens count
        max_length = tf.reduce_max(inputs_length)

        inputs_embedding = tf.nn.embedding_lookup(self.token_embeddings, token_ids)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            self._rnn_cell(training),
            self._rnn_cell(training),
            inputs_embedding,
            sequence_length=inputs_length,
            dtype=tf.float32
        )

        rnn_output = tf.concat([output_fw, output_bw], 0)

        rnn_output = tf.reshape(rnn_output, [-1, self.param_rnn_size * 2])
        softmax_logits = tf.nn.xw_plus_b(rnn_output, self.softmax_weight, self.softmax_bias)
        softmax_logits = tf.reshape(softmax_logits, [-1, max_length, Model.NUM_CLASSES])

        return softmax_logits


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = Model(params['vocab_words'], params['embed_size'], params['rnn_layers'], params['rnn_size'],
                  params['keep_prob'])

    inputs = features
    if isinstance(inputs, dict):
        inputs = features['inputs']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(inputs, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            # export_outputs={
            #     'classify': tf.estimator.export.PredictOutput(predictions)
            # }
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(inputs, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metrics = {
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(logits, axis=2),
                # weights=params['class_weights']
            ),
            'f1_score': f1_score(
                labels=labels,
                predictions=tf.argmax(logits, axis=2),
                # weights=params['class_weights']
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metrics
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(inputs, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits
                                                      # , weights=params['class_weights']
                                                      )
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.argmax(logits, axis=2),
            # weights=params['class_weights']
        )
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train
        )
