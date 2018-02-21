from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import tensorflow as tf
from io import open
from operator import itemgetter
from random import Random
from functools import partial
from six.moves import cPickle
from tfucops import expand_split_words

from .vocab import Vocabulary

PAD_TOKEN = '<PAD>'


class Trainer:
    DATASET_SOURCE_NAME = 'dataset.txt'
    DATASET_TARGET_NAME = 'dataset.pkl'
    VOCABULARY_MIN_FREQ = 10
    VOCABULARY_TARGET_NAME = 'vocabulary.pkl'
    WEIGHTS_TARGET_NAME = 'weights.pkl'

    def __init__(self, data_dir, test_size, batch_size, doc_size, random_seed=None):
        self.data_dir = data_dir

        self.test_size = test_size
        assert 0 < test_size < 1, 'test_size should be between (0; 1)'

        self.batch_size = batch_size
        assert 0 < batch_size, 'batch_size should be above 0'

        self.doc_size = doc_size
        assert 0 < doc_size, 'doc_size should be above 0'

        self._random_generator = Random()
        self._random_generator.seed(random_seed)

        self._load_dataset()
        self._split_dataset()
        self._load_vocab()
        self._load_weights()

    def _load_dataset(self):
        dataset_filename = os.path.join(self.data_dir, Trainer.DATASET_TARGET_NAME)

        if not os.path.exists(dataset_filename):
            self._prepare_dataset()

        tf.logging.info('Loading dataset from {}'.format(dataset_filename))
        with open(dataset_filename, 'rb') as fin:
            self._full_data = cPickle.load(fin)
            tf.logging.info('Dataset loaded from {}'.format(dataset_filename))

    def _prepare_dataset(self):
        source_filename = os.path.join(self.data_dir, Trainer.DATASET_SOURCE_NAME)
        tf.logging.info('Creating dataset from {}'.format(source_filename))

        input_ph = tf.placeholder(tf.string, shape=(None))
        split_op = split_tokens(input_ph)
        result_op = tf.sparse_tensor_to_dense(split_op, default_value='')

        with open(source_filename, 'r', encoding='utf-8') as dataset_file:
            paragraphs = dataset_file.read()
        paragraphs = paragraphs.split('\n\n')  # 0D -> 1D (text -> paragraphs)
        paragraphs = map(lambda p: p.split('\n'), paragraphs)  # 1D -> 2D (paragraphs -> sentences)
        paragraphs = map(partial(map, lambda s: s.strip()), paragraphs)  # strip sentences
        paragraphs = map(partial(filter, len), paragraphs)  # filter out 0-len sentences
        paragraphs = map(list, paragraphs)  # sentence iterator -> sentence list
        paragraphs = filter(len, paragraphs)  # filter out 0-len paragraphs
        paragraphs = list(paragraphs)  # paragraph iterator -> paragraph list

        new_paragraphs = []
        with tf.Session() as sess:
            for sentences in paragraphs:
                tokens = sess.run(result_op, feed_dict={input_ph: sentences})
                tokens = map(partial(filter, len), tokens)  # filter out 0-len tokens
                tokens = map(partial(map, lambda t: t.decode('utf-8')), tokens)  # decode binary tokens as UTF-8
                tokens = list(map(list, tokens))  # tokens iterator -> tokens list

                new_paragraphs.append(tokens)

        dataset_filename = os.path.join(self.data_dir, Trainer.DATASET_TARGET_NAME)
        with open(dataset_filename, 'wb') as fout:
            cPickle.dump(new_paragraphs, fout, protocol=2)
        tf.logging.info('Dataset saved to {}'.format(dataset_filename))

    def _split_dataset(self):
        assert isinstance(self._full_data, list)

        split_size = int(round(len(self._full_data) * self.test_size))
        self._test_data = self._full_data[:split_size]  # take from head due to complex ending of dataset
        self._train_data = self._full_data[split_size:]
        self._full_data = None

    def _load_vocab(self):
        vocab_filename = os.path.join(self.data_dir, Trainer.VOCABULARY_TARGET_NAME)
        if not os.path.exists(vocab_filename):
            self._prepare_vocab()

        tf.logging.info('Loading vocabulary from {}'.format(vocab_filename))
        self._train_vocab = Vocabulary.load(vocab_filename)
        tf.logging.info('Vocabulary loaded from {}'.format(vocab_filename))
        self._train_vocab.save(vocab_filename[:-4] + '.txt', False)

    def _prepare_vocab(self):
        assert isinstance(self._train_data, list)

        tf.logging.info('Creating vocabulary')
        words = itertools.chain.from_iterable(self._train_data)  # list of paragraphs to list of sentences
        words = itertools.chain.from_iterable(words)  # list of sentences to list of tokens
        words = list(words)

        vocab_filename = os.path.join(self.data_dir, Trainer.VOCABULARY_TARGET_NAME)
        vocab = Vocabulary()
        vocab.fit(words)
        vocab.trim(Trainer.VOCABULARY_MIN_FREQ)
        vocab.save(vocab_filename)
        tf.logging.info('Vocabulary (as binary) saved to {}'.format(vocab_filename))
        vocab.save(vocab_filename[:-4] + '.txt', False)
        tf.logging.info('Vocabulary (as text) saved to {}'.format(vocab_filename[:-4] + '.txt'))

    def vocab_words(self):
        assert isinstance(self._train_vocab, Vocabulary)

        return self._train_vocab.items()

    def _load_weights(self):
        weights_filename = os.path.join(self.data_dir, Trainer.WEIGHTS_TARGET_NAME)
        if not os.path.exists(weights_filename):
            self._prepare_weights()

        tf.logging.info('Loading weights from {}'.format(weights_filename))
        with open(weights_filename, 'rb') as fin:
            self._train_weights = cPickle.load(fin)
            tf.logging.info('Weights loaded from {}'.format(weights_filename))

    def _prepare_weights(self):
        assert isinstance(self._train_data, list)

        tf.logging.info('Creating weights')
        counts = itertools.chain.from_iterable(self._train_data)  # list of paragraphs to list of sentences
        counts = map(list, counts)  # extract token counts from sentence tuple
        counts = map(len, counts)  # extract token counts from sentence tuple
        counts = list(counts)

        counts_0 = sum(counts)
        counts_1 = len(counts)
        weight_0 = counts_0 / (counts_0 + counts_1)

        weights = [[1 - weight_0], [weight_0]]

        weights_filename = os.path.join(self.data_dir, Trainer.WEIGHTS_TARGET_NAME)
        with open(weights_filename, 'wb') as fout:
            cPickle.dump(weights, fout, protocol=2)
        tf.logging.info('Weights saved to {}'.format(weights_filename))

    def train_weights(self):
        assert isinstance(self._train_weights, list)

        return self._train_weights

    def _data_generator(self, data):
        assert isinstance(data, list)

        sample_used = 0
        sample_size = self._random_generator.randint(1, self.doc_size)

        while len(data) > 0:
            if sample_used == self.batch_size:
                sample_used = 0
                sample_size = self._random_generator.randint(1, self.doc_size)

            sample = data[:sample_size]
            data = data[sample_size:]

            sample = list(itertools.chain.from_iterable(sample))  # 3-D list of sentences to 2-D
            sample_used += 1

            X = []
            y = []
            X_glue = [' ']
            y_glue = [1]

            for tokens in sample:
                X.extend(tokens + X_glue)
                y.extend([0] * len(tokens) + y_glue)
            assert len(X) == len(y), 'items count should be equal labels count'

            yield u''.join(X[:-1]), y[:-1]

    def _train_generator(self):
        data = list(self._train_data)  # make a copy
        self._random_generator.shuffle(data, self._random_generator.random)

        return self._data_generator(data)

    def _test_generator(self):
        return self._data_generator(self._test_data)

    def _dataset_generator(self, generator):
        dataset = tf.data.Dataset.from_generator(
            generator,
            (tf.string, tf.int32),
            (tf.TensorShape([]), tf.TensorShape([None]))
        )
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=([], [None]))
        # dataset = dataset.map(lambda X, y: ({'document': X}, y))
        dataset = dataset.repeat(4)
        dataset = dataset.prefetch(4)

        return dataset

    def train_dataset(self):
        return self._dataset_generator(self._train_generator)

    def test_dataset(self):
        return self._dataset_generator(self._test_generator)


def predict_input_fn(documents, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(documents)
    dataset = dataset.map(lambda x: expand_split_words(x, default_value=PAD_TOKEN)) # TODO: parallel
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None]), padding_values=(PAD_TOKEN))

    return dataset
