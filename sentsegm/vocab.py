from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from io import open
from operator import itemgetter
from six.moves import cPickle


class Vocabulary:
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    def __init__(self):
        self._cnt = Counter()

    def fit(self, items):
        assert isinstance(items, list), 'items should be a list'
        self._cnt.update(items)

    def trim(self, min_freq):
        for word in self.items():
            if self._cnt[word] < min_freq:
                del self._cnt[word]

    def items(self):
        # Due to different behaviour for items with same counts in Python 2 and 3 we should resort result
        result = self._cnt.most_common()
        result.sort(key=itemgetter(0))
        result.sort(key=itemgetter(1), reverse=True)
        result, _ = zip(*result)

        return list(result)

    def save(self, filename, binary=True):
        if binary:
            with open(filename, 'wb') as fout:
                cPickle.dump(self._cnt, fout, protocol=2)
        else:
            with open(filename, 'w', encoding='utf-8') as fout:
                fout.write(u'token\tfrequency\n')
                for w in self.items():
                    f = self._cnt[w]
                    w_safe = w if not w.isspace() else 'space_{}'.format(ord(w))
                    w_safe = w if len(w_safe) else '<EMPTY>'
                    fout.write(u'{}\t{}\n'.format(w_safe, f))

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fin:
            cnt = cPickle.load(fin)

        inst = Vocabulary()
        inst._cnt = cnt

        return inst
