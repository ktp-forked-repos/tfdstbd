from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .metric import f1_score


class TokenRNN:
    def __init__(self):
        self.mode = None
        self.params = None
        self.input_data = None
        self.targets = None

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params

        self.input_data = features
        if type(features) == dict:
            self.input_data = features['input_data']
        self.targets = labels

        self._build_graph()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'probs': self.probs}
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=self.loss,
            train_op=self.train_op
        )

    def _build_graph(self):
        self._create_embedding()
        self._create_rnn_cells()
        self._create_inferece()
        self._create_predictions()

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._create_loss()
            self._creat_train_op()

    def _create_embedding(self):
        self.embeddings = tf.get_variable('embeddings', [self.params['vocab_size'], self.params['embed_size']])

    def _create_rnn_cells(self):
        cells = [self._create_rnn_cell() for _ in range(self.params['num_layers'])]
        self.rnn_cells = tf.contrib.rnn.MultiRNNCell(cells)
        self.initial_state = self.rnn_cells.zero_state(self.params['batch_size'])

    def _create_rnn_cell(self):
        cell = tf.contrib.rnn.GRUCell(self.params['rnn_size'])
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.params['input_keep_prob'],
                                                 output_keep_prob=self.params['output_keep_prob'])
        return cell

    def _create_inferece(self):
        length = tf.argmax(self.targets)

        input_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_data)
        if self.mode == tf.estimator.ModeKeys.TRAIN and self.params['output_keep_prob']:
            input_embeddings = tf.nn.dropout(input_embeddings, self.params['output_keep_prob'])

        output, state = tf.nn.dynamic_rnn(self.rnn_cells, input_embeddings, initial_state=self.initial_state)

        output = tf.reshape(output, [-1, self.params['rnn_size']])
        softmax_w = tf.get_variable('softmax_w', [self.params['rnn_size???_SEQ_LEN?'], self.params['vocab_size']])
        softmax_b = tf.get_variable('softmax_b', [self.params['vocab_size']])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        self.logits = tf.reshape(logits,
                                 [self.params['batch_size'], self.params['rnn_size'], self.params['vocab_size']])
        self.probs = tf.nn.softmax(self.logits, name='probs')

    def _create_predictions(self):
        self.predictions = tf.argmax(self.probs, axis=1)
        tf.identity(self.predictions[:self.params['seq_length']], 'prediction_0')

    def _create_loss(self):
        # Use the contrib sequence loss and average over the batches
        mask = tf.ones([self.params['batch_size'], self.params['rnn_size']])  # TODO: sequence_mask
        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,  # [batch_size, sequence_length, num_decoder_symbols???]
            self.targets,  # [batch_size, sequence_length]
            mask,
            average_across_timesteps=False)
        self.loss = tf.reduce_sum(loss)

    def _creat_train_op(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=self.params['learning_rate']
        )


class Model:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    NUM_CLASSES = 2

    def __init__(self, vocab_words, embed_size, rnn_size):
        # PAD_TOKEN should have index==0 to calculate real sequence length later in padded batch
        vocab_words = [Model.PAD_TOKEN, Model.UNK_TOKEN] + vocab_words
        vocab_size = len(vocab_words)
        self.vocab_words = tf.constant(vocab_words)
        self.vocab_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=self.vocab_words,
            num_oov_buckets=0,
            default_value=1  # UNK_TOKEN
        )

        self.param_rnn_size = rnn_size

        self.token_embeddings = tf.get_variable('token_embeddings', [vocab_size, embed_size])
        self.softmax_weight = tf.get_variable('softmax_weight', [self.param_rnn_size, Model.NUM_CLASSES])
        self.softmax_bias = tf.get_variable('softmax_bias', [Model.NUM_CLASSES])

    def __call__(self, inputs, training):
        """Add operations to classify a batch of tokens.
        Args:
          inputs: A Tensor representing a batch of tokenized documents.
          training: A boolean. Set to True to add operations required only when training the classifier.
        Returns:
          A logits Tensor with shape [<batch_size>, 2].
        """

        inputs_id = self.vocab_table.lookup(inputs)  # tokens -> ids
        inputs_mask = tf.sign(inputs_id)  # 0 for padded tokens, 1 for real ones
        inputs_length = tf.cast(tf.reduce_sum(inputs_mask, 1), tf.int32)  # real tokens count
        batch_size, max_length = tf.unstack(tf.shape(inputs)) # TODO: check it

        inputs_embedding = tf.nn.embedding_lookup(self.token_embeddings, inputs_id)
        rnn_output, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self.param_rnn_size),
            inputs_embedding,
            sequence_length=inputs_length,
            dtype=tf.float32
        )

        rnn_output = tf.reshape(rnn_output, [-1, self.param_rnn_size])
        softmax_logits = tf.nn.xw_plus_b(rnn_output, self.softmax_weight, self.softmax_bias)
        softmax_logits = tf.reshape(softmax_logits, [-1, max_length, Model.NUM_CLASSES])

        return softmax_logits


# def f1_score(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
#     with tf.variable_scope(name, 'f1_score'):
#         precision = tf.metrics.precision(
#             labels,
#             predictions,
#             weights,
#             metrics_collections,
#             updates_collections,
#             'precision'
#         )
#         recall = tf.metrics.recall(
#             labels,
#             predictions,
#             weights,
#             metrics_collections,
#             updates_collections,
#             'recall'
#         )
#
#     return 2 * precision * recall / (precision + recall)


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = Model(params['vocab_words'], params['embed_size'], params['rnn_size'])

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
                predictions=tf.argmax(logits, axis=2)
            ),
            'f1_score': f1_score(
                labels=labels,
                predictions=tf.argmax(logits, axis=2)
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metrics
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(inputs, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train = optimizer.minimize(loss, tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train
        )
