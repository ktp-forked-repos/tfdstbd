from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.contrib.training import HParams
from .model import sbd_model_fn


class SBDEstimator(estimator.Estimator):
    def __init__(self,
                 min_n,  # minimum ngram length
                 max_n,  # maximum ngram length
                 ngram_vocab,  # list of vocabulary ngrams
                 uniq_count,  # number of uniq ngrams buckets
                 embed_size,  # size of char embedding
                 rnn_size,  # size of single RNN layer
                 rnn_layers,  # number of RNN layers
                 use_cudnn,  # use Cudnn LSTM
                 keep_prob,  # 1 - dropout probability
                 learning_rate,  # learning rate
                 model_dir=None,
                 config=None):
        params = HParams(
            min_n=min_n,
            max_n=max_n,
            ngram_vocab=ngram_vocab,
            uniq_count=uniq_count,
            embed_size=embed_size,
            rnn_size=rnn_size,
            rnn_layers=rnn_layers,
            use_cudnn=use_cudnn,
            keep_prob=keep_prob,
            learning_rate=learning_rate,
        )

        super(SBDEstimator, self).__init__(
            model_fn=sbd_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
        )
