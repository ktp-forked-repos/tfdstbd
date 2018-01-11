from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import open
import os
import random
import itertools

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer


# try:
#     from functools import lru_cache
# except:
#     from repoze.lru import lru_cache


class TrainDataLoader:
    test_sentences = None
    train_sentences = None
    word_tokenizer = None
    word_detokenizer = None

    def __init__(self, data_dir, batch_size=32, test_size=0.2, max_docsize=100):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.test_size = test_size
        assert test_size <= 0 or test_size >= 1, 'test_size mus be between (0; 1)'

        self.max_docsize = max_docsize

        self.word_tokenizer = TreebankWordTokenizer()
        self.word_detokenizer = TreebankWordDetokenizer()

    def _load_sentences(self):
        if self.test_sentences is not None and self.train_sentences is not None:
            return

        dataset_filename = os.path.join(self.data_dir, 'sentencies.txt')
        with open(dataset_filename, encoding='utf-8') as dataset_file:
            paragraphs = dataset_file.read().split('\n\n')

        all_sentences = []
        for paragraph in paragraphs:
            sentences = [s.strip() for s in paragraph.split('\n') if len(s.strip()) > 0]
            if len(sentences) > 0:
                all_sentences.append(sentences)

        split_size = int(round(len(all_sentences) * self.test_size))
        self.test_sentences = all_sentences[:split_size]  # take from head due to complex end of dataset
        self.train_sentences = all_sentences[split_size:]

    def _train_generator(self):
        self._load_sentences()

        return self._data_generator(self.train_sentences)

    def _test_generator(self):
        self._load_sentences()

        return self._data_generator(self.test_sentences)

    def _data_generator(self, sentences):
        random.shuffle(sentences)

        sample_used = 0
        sample_size = random.randint(1, self.max_docsize)

        while len(sentences) > 0:

            if sample_used == self.batch_size:
                sample_used = 0
                sample_size = random.randint(1, self.max_docsize)

            X = []
            y = []

            sample = sentences[:sample_size]
            sentences = sentences[sample_size:]

            sample = list(itertools.chain.from_iterable(sample))  # 2-D list of sentences to 1-D

            for sentence in sample:
                words = self.word_tokenizer.tokenize(sentence, return_str=True).split(' ')
                X.extend(words)
                y.extend([0] * (len(words) - 1))
                y.append(1)

            assert len(X) == len(y), 'words count should be equal labels count'

            yield X, y

            # @lru_cache(maxsize=10000)
            # def _split_words(self, sentence):
            #     return self.word_tokenizer.tokenize(sentence)
            # vocab_file = os.path.join(data_dir, 'vocab.pkl')

            # if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            #     print('reading text file')
            #     self.preprocess(input_file, vocab_file, tensor_file)
            # else:
            #     print('loading preprocessed files')
            #     self.load_preprocessed(vocab_file, tensor_file)

            # def preprocess(self, input_file, vocab_file, tensor_file):
            #     with codecs.open(input_file, 'r', encoding='utf-8') as f:
            #         data = f.read()
            #     counter = collections.Counter(data)
            #     count_pairs = sorted(counter.items(), key=lambda x: -x[1])
            #     self.chars, _ = zip(*count_pairs)
            #     self.vocab_size = len(self.chars)
            #     self.vocab = dict(zip(self.chars, range(len(self.chars))))
            #     with open(vocab_file, 'wb') as f:
            #         cPickle.dump(self.chars, f)
            #     self.tensor = np.array(list(map(self.vocab.get, data)))
            #     np.save(tensor_file, self.tensor)
            #
            # def load_preprocessed(self, vocab_file, tensor_file):
            #     with open(vocab_file, 'rb') as f:
            #         self.chars = cPickle.load(f)
            #     self.vocab_size = len(self.chars)
            #     self.vocab = dict(zip(self.chars, range(len(self.chars))))
            #     self.tensor = np.load(tensor_file)
            #
            # def vocab_size(self):
            #     return self.vocab_size
            #
            # def make_train_and_test_set(self, train_size=0.8, test_size=0.2):
            #     self.num_batches = int(self.tensor.size / (self.batch_size *
            #                                                self.seq_length))
            #
            #     # When the data (tensor) is too small,
            #     # let's give them a better error message
            #     if self.num_batches == 0:
            #         assert False, 'Not enough data. Make seq_length and batch_size small.'
            #     if train_size + test_size > 1:
            #         assert False, 'train_size and test_size are large. sum > 1'
            #
            #     self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
            #     xdata = self.tensor
            #     ydata = np.copy(self.tensor)
            #     ydata[:-1] = xdata[1:]
            #     ydata[-1] = xdata[0]
            #
            #     self.X = xdata
            #     self.y = ydata
            #
            #     train_length = int(len(self.X) / self.seq_length * train_size) * self.seq_length
            #     test_length = int(len(self.X) / self.seq_length * test_size) * self.seq_length
            #
            #     train_X = self.X[train_length:]
            #     train_y = self.y[train_length:]
            #
            #     test_X = self.X[:test_length]
            #     test_y = self.y[:test_length]
            #
            #     return train_X, test_X, train_y, test_y
            #
            # def create_batches(self):
            #     self.num_batches = int(self.tensor.size / (self.batch_size *
            #                                                self.seq_length))
            #
            #     self.X_batches = np.split(self.X.reshape(self.batch_size, -1),
            #                               self.num_batches, 1)
            #     self.y_batches = np.split(self.y.reshape(self.batch_size, -1),
            #                               self.num_batches, 1)
            #     self.reset_batch_pointer()
            #
            # def next_batch(self):
            #     X, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
            #     self.pointer += 1
            #     return X, y
            #
            # def reset_batch_pointer(self):
            #     self.pointer = 0
