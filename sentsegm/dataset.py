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
from .ops import split_tokens

try:
    # Python 2
    from future_builtins import filter, map
except ImportError:
    # Python 3
    pass

from .vocab import Vocabulary


PAD_TOKEN = '<PAD>'

class Trainer:
    DATASET_SOURCE_NAME = 'dataset.txt'
    DATASET_TARGET_NAME = 'dataset.pkl'
    VOCABULARY_MIN_FREQ = 10
    VOCABULARY_TARGET_NAME = 'vocabulary.pkl'
    WEIGHTS_TARGET_NAME = 'weights.pkl'

    def __init__(self, data_dir, test_size, batch_size, doc_size, random_seed=None):
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)

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

        print('Loading dataset from {}'.format(dataset_filename))
        with open(dataset_filename, 'rb') as fin:
            self._full_dataset = cPickle.load(fin)
            print('Dataset loaded from {}'.format(dataset_filename))

    def _prepare_dataset(self):
        source_filename = os.path.join(self.data_dir, Trainer.DATASET_SOURCE_NAME)
        print('Creating dataset from {}'.format(source_filename))

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
        paragraphs = list(paragraphs) # paragraph iterator -> paragraph list

        new_paragraphs = []
        with tf.Session() as sess:
            for sentences in paragraphs:
                tokens = sess.run(result_op, feed_dict={input_ph: sentences})
                tokens = map(partial(map, lambda t: t.decode('utf-8')), tokens)  # decode binary tokens as UTF-8
                tokens = list(map(list, tokens))  # tokens iterator -> tokens list

                counts = map(partial(filter, len), tokens)  # filter out 0-len tokens
                counts = map(partial(map, list), counts)  # token iterator -> token list
                counts = list(map(list, counts))  # tokens iterator -> tokens list
                counts = list(map(len, counts))

                new_sentences = list(zip(sentences, tokens, counts))
                new_paragraphs.append(new_sentences)

        dataset_filename = os.path.join(self.data_dir, Trainer.DATASET_TARGET_NAME)
        with open(dataset_filename, 'wb') as fout:
            cPickle.dump(new_paragraphs, fout, protocol=2)
        print('Dataset saved to {}'.format(dataset_filename))

    def _split_dataset(self):
        assert isinstance(self._full_dataset, list)

        split_size = int(round(len(self._full_dataset) * self.test_size))
        self._test_dataset = self._full_dataset[:split_size]  # take from head due to complex ending of dataset
        self._train_dataset = self._full_dataset[split_size:]
        self._full_dataset = None

    def _load_vocab(self):
        vocab_filename = os.path.join(self.data_dir, Trainer.VOCABULARY_TARGET_NAME)
        if not os.path.exists(vocab_filename):
            self._prepare_vocab()

        print('Loading vocabulary from {}'.format(vocab_filename))
        self._train_vocab = Vocabulary.load(vocab_filename)
        print('Vocabulary loaded from {}'.format(vocab_filename))

    def _prepare_vocab(self):
        assert isinstance(self._train_dataset, list)

        print('Creating vocabulary')
        words = itertools.chain.from_iterable(self._train_dataset)  # list of paragraphs to list of sentences
        words = map(itemgetter(1), words) # extract tokens from sentence tuple
        words = itertools.chain.from_iterable(words)  # 2-D list of sentences to 1-D list of tokens
        words = list(words)

        vocab_filename = os.path.join(self.data_dir, Trainer.VOCABULARY_TARGET_NAME)
        vocab = Vocabulary()
        vocab.fit(words)
        vocab.trim(Trainer.VOCABULARY_MIN_FREQ)
        vocab.save(vocab_filename)
        print('Vocabulary saved to {}'.format(vocab_filename))

    def vocab_words(self):
        assert isinstance(self._train_vocab, Vocabulary)

        return self._train_vocab.items()

    def _load_weights(self):
        weights_filename = os.path.join(self.data_dir, Trainer.WEIGHTS_TARGET_NAME)
        if not os.path.exists(weights_filename):
            self._prepare_weights()

        print('Loading weights from {}'.format(weights_filename))
        with open(weights_filename, 'rb') as fin:
            self._train_weights = cPickle.load(fin)
            print('Weights loaded from {}'.format(weights_filename))

    def _prepare_weights(self):
        assert isinstance(self._train_dataset, list)

        print('Creating weights')
        counts = itertools.chain.from_iterable(self._train_dataset)  # list of paragraphs to list of sentences
        counts = map(itemgetter(2), counts) # extract token counts from sentence tuple
        counts = list(counts)

        counts_0 = sum(counts)
        counts_1 = len(counts)
        weight_0 = counts_0 / (counts_0 + counts_1)

        weights = [1 - weight_0, weight_0]

        weights_filename = os.path.join(self.data_dir, Trainer.WEIGHTS_TARGET_NAME)
        with open(weights_filename, 'wb') as fout:
            cPickle.dump(weights, fout, protocol=2)
        print('Weights saved to {}'.format(weights_filename))

    def _train_generator(self):
        data = list(self._train_dataset)  # make a copy
        self._random_generator.shuffle(data, self._random_generator.random)

        return self._data_generator(data)

    def _test_generator(self):
        return self._data_generator(self._test_dataset)

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

            for sentence in sample:
                tokens = sentence

                X.extend(tokens + X_glue)
                y.extend([0] * len(tokens) + y_glue)
            assert len(X) == len(y), 'items count should be equal labels count'

            yield X, y

    def train_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self._train_generator,
            (tf.string, tf.int32),
            (tf.TensorShape([None]), tf.TensorShape([None]))
        )
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=([None], [None]), padding_values=(PAD_TOKEN, 0))
        dataset = dataset.prefetch(4)

        return dataset

    def test_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self._test_generator,
            (tf.string, tf.int32),
            (tf.TensorShape([None]), tf.TensorShape([None]))
        )
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=([None], [None]), padding_values=(PAD_TOKEN, 0))
        # dataset = dataset.prefetch(4)

        return dataset
