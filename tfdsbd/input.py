from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import tensorflow as tf
from random import Random
from functools import partial
from six.moves import cPickle
from tfucops import expand_split_words, transform_normalize_unicode

from .vocab import Vocabulary
from .model import Model


class Trainer:
    DATASET_SOURCE_NAME = 'dataset.txt'
    DATASET_TARGET_NAME = 'dataset.pkl'
    VOCABULARY_MIN_FREQ = 10
    VOCABULARY_TARGET_NAME = 'vocabulary.pkl'
    WEIGHTS_TARGET_NAME = 'weights.pkl'

    def __init__(self, data_dir, test_size, batch_size, doc_size):
        self.data_dir = data_dir

        self.test_size = test_size
        assert 0 < test_size < 1, 'test_size should be between (0; 1)'

        self.batch_size = batch_size
        assert 0 < batch_size, 'batch_size should be above 0'

        self.doc_size = doc_size
        assert 0 < doc_size, 'doc_size should be above 0'

        self._random_generator = Random()

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
        result_op = transform_normalize_unicode(input_ph, 'NFC')
        result_op = expand_split_words(result_op)

        with open(source_filename, 'rb') as dataset_file:
            paragraphs = dataset_file.read().decode('utf-8')
        paragraphs = paragraphs.split('\n\n')  # 0D -> 1D (text -> paragraphs)
        paragraphs = map(lambda p: p.split('\n'), paragraphs)  # 1D -> 2D (paragraphs -> sentences)
        paragraphs = map(partial(map, lambda s: s.strip('\n')), paragraphs)  # strip sentences
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

        # Hack to save PAD & UNK TOKENS in same order as Model use them in embedding vocabulary
        _, max_freq = vocab.most_common(1)[0]
        vocab.fit([Model.PAD_TOKEN] * (max_freq + 2))
        vocab.fit([Model.UNK_TOKEN] * (max_freq + 1))
        vocab.save(vocab_filename[:-4] + '.tsv', False)
        tf.logging.info('Vocabulary (as TSV) saved to {}'.format(vocab_filename[:-4] + '.txt'))

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

            X = [_.encode('utf-8') for _ in X]  # required due to TF 1.6.0rc1 bug in Python2

            yield {'document': X[:-1]}, y[:-1]

    def _train_generator(self):
        print('TRAIN GENERATOR // TRAIN GENERATOR // TRAIN GENERATOR')
        data = list(self._train_data)  # make a copy
        self._random_generator.shuffle(data, self._random_generator.random)

        return self._data_generator(data)

    def _test_generator(self):
        print('TEST GENERATOR // TEST GENERATOR // TEST GENERATOR')
        return self._data_generator(self._test_data)

    def _dataset_generator(self, generator):
        dataset = tf.data.Dataset.from_generator(
            generator,
            ({'document': tf.string}, tf.int32),
            ({'document': tf.TensorShape([None])}, tf.TensorShape([None]))
        )
        # dataset = dataset.map(_transform_form, num_parallel_calls=10) # TODO
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=({'document': [None]}, [None]),
            padding_values=({'document': Model.PAD_TOKEN}, 0))
        dataset = dataset.map(_extend_length, num_parallel_calls=10)

        return dataset

    def train_input_fn(self):
        dataset = self._dataset_generator(self._train_generator)
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat(50)
        dataset = dataset.prefetch(10)

        return dataset

    def test_input_fn(self):
        return self._dataset_generator(self._test_generator)


def train_input_fn(wildcard, batch_size):
    # Create dataset from multiple TFRecords files
    files = tf.data.TFRecordDataset.list_files(wildcard)
    dataset = files.interleave(
        lambda file: tf.data.TFRecordDataset(file, compression_type='GZIP'),
        cycle_length=5
    )

    # Parse serialized examples
    def _parse_example(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'document': tf.FixedLenFeature(1, tf.string),
                'labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            })

        return {'document': features['document'][0]}, features['labels']
        # context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        #     serialized=example_proto,
        #     context_features={'document': tf.FixedLenFeature([], dtype=tf.string)},
        #     sequence_features={'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        # )
        # return context_parsed['document'], sequence_parsed['labels']

    dataset = dataset.map(_parse_example)

    # Extract features
    def _parse_features(features, labels):
        features['tokens'] = expand_split_words(
            features['document'],
            default_value=Model.PAD_TOKEN
        )
        features['length'] = tf.size(features['tokens'])

        return features, labels

    dataset = dataset.map(_parse_features, num_parallel_calls=10)

    # Create padded batch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=({'document': 1, 'tokens': [None], 'length': 1}, [None]),
        padding_values=({'tokens': Model.PAD_TOKEN}, 0)
    )
    dataset = dataset.prefetch(10)

    return dataset


# def predict_input_fn(documents, batch_size=1):
#     assert type(documents) == list
#
#     expected_filename = os.path.join(self.temp_dir, '*.tfrecords.gz')
#     files = tf.data.TFRecordDataset.list_files(expected_filename)
#     dataset = files.interleave(
#         lambda f: tf.data.TFRecordDataset([f], compression_type='GZIP'),
#         cycle_length=5
#     )
#     dataset = dataset.map(_parse_function)
#     iterator = dataset.make_one_shot_iterator()
#     next_element = iterator.get_next()
#
#     dataset = tf.data.Dataset.from_tensor_slices({'document': documents})
#
#     # dataset = dataset.map(_transform_form, num_parallel_calls=10) # TODO
#     dataset = dataset.map(_split_words, num_parallel_calls=10)
#     dataset = dataset.padded_batch(
#         batch_size,
#         padded_shapes={'document': [None]},
#         padding_values={'document': Model.PAD_TOKEN})
#     dataset = dataset.prefetch(10)
#
#     return dataset

#
# def _split_words(features):
#     assert type(features) == dict
#     assert 'document' in features
#
#     features['document'] = expand_split_words(features['document'], default_value=Model.PAD_TOKEN)
#
#     return features
#
#
# def _transform_form(*args):
#     assert len(args) > 0
#
#     assert type(args[0]) == dict
#     assert 'document' in args[0]
#
#     args[0]['document'] = transform_normalize_unicode(args[0]['document'], 'NFC')
#
#     return args
#
#
# def _extend_length(*args):
#     assert len(args) > 0
#
#     assert type(args[0]) == dict
#     assert 'document' in args[0]
#
#     mask = tf.not_equal(args[0]['document'], Model.PAD_TOKEN)  # 0 for padded tokens, 1 for real ones
#     mask = tf.cast(mask, tf.int32)
#     args[0]['length'] = tf.reduce_sum(mask, 1)  # real tokens count
#
#     return args
