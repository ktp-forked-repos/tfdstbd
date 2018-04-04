from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from io import open
from collections import Counter
from operator import itemgetter
from six.moves import cPickle


class Vocabulary:
    def __init__(self):
        self._cnt = Counter()

    def fit(self, items):
        assert isinstance(items, list), 'items should be a list'
        self._cnt.update(items)

    def trim(self, min_freq):
        for word in list(self._cnt.keys()):
            if self._cnt[word] < min_freq:
                del self._cnt[word]

    def items(self):
        # Due to different behaviour for items with same counts in Python 2 and 3 we should resort result ourselves
        result = self._cnt.most_common()
        result.sort(key=itemgetter(0))
        result.sort(key=itemgetter(1), reverse=True)
        result, _ = zip(*result)

        return list(result)

    def most_common(self, n=None):
        return self._cnt.most_common(n)

    def save(self, filename):
        with open(filename, 'wb') as fout:
            cPickle.dump(self._cnt, fout, protocol=2)

    def export(self, filename, header=True):
        def _safe(word):
            word = u'{}'.format(word)
            if not len(word): return '[]'
            word = re.sub(r'\s', lambda match: '[{}]'.format(ord(match.group(0))), word)
            return word

        with open(filename, 'w', encoding='utf-8') as fout:
            if header:
                line = u'{}\t{}\n'.format(_safe('token'), _safe('frequency'))
                fout.write(line)
            for w in self.items():
                line = u'{}\t{}\n'.format(_safe(w.decode('utf-8')), _safe(self._cnt[w]))
                fout.write(line)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fin:
            cnt = cPickle.load(fin)

        inst = Vocabulary()
        inst._cnt = cnt

        return inst
