from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from .dataset import Trainer
from .model import model_fn
from .params import model_params


# CompiledWrapper
# https://www.tensorflow.org/api_guides/python/contrib.seq2seq

# TODO
# features, corpus https://nlp.stanford.edu/courses/cs224n/2005/agarwal_herndon_shneider_final.pdf
# http://www.aclweb.org/anthology/C12-2096
# http://amitavadas.com/Pub/SBD_ICON_2015.pdf
# http://www.wellformedness.com/blog/simpler-sentence-boundary-detection/ - contain vowel
# http://www.aclweb.org/anthology/D13-1146
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.5017&rep=rep1&type=pdf

# translit http://userguide.icu-project.org/transforms/general

tf.logging.set_verbosity(tf.logging.INFO)

trainer_params = {
    'data_dir': os.path.join(os.path.dirname(__file__), 'data'),
    # 'data_dir': 'data_s',
    'batch_size': 100,
    'test_size': 0.2,
    'doc_size': 100,
    'random_seed': 43,
}
trainer = Trainer(**trainer_params)

params = model_params()
params['vocab_words'] = trainer.vocab_words()
params['class_weights'] = trainer.train_weights()

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='../model',
    params=params
)

for i in range(100):
    estimator.train(input_fn=trainer.train_dataset)

    if i % 10:
        metrics = estimator.evaluate(input_fn=trainer.test_dataset)
        print(metrics)

