from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def model_params():
    return {
        # 'vocab_words': trainer.vocab_words(),
        'embed_size': 50,
        'rnn_layers': 2,
        'rnn_size': 128,
        'keep_prob': 0.8
    }
