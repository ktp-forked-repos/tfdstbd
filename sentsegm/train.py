from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .dataset import Trainer
from .model import model_fn


# https://www.tensorflow.org/api_docs/python/tf/parse_single_sequence_example
# https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
# CompiledWrapper
# https://www.tensorflow.org/api_guides/python/contrib.seq2seq

# TODO
# features, corpus https://nlp.stanford.edu/courses/cs224n/2005/agarwal_herndon_shneider_final.pdf
# http://www.aclweb.org/anthology/C12-2096
# http://amitavadas.com/Pub/SBD_ICON_2015.pdf
# http://www.wellformedness.com/blog/simpler-sentence-boundary-detection/ - contain vowel
# http://www.aclweb.org/anthology/D13-1146

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.5017&rep=rep1&type=pdf

# TODO http://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt

# translit http://userguide.icu-project.org/transforms/general

tf.logging.set_verbosity(tf.logging.DEBUG)

trainer_params = {
    'data_dir': 'data',
    # 'data_dir': 'data_s',
    'batch_size': 20,
    'test_size': 0.2,
    'doc_size': 10,
    'random_seed': 43,
}
trainer = Trainer(**trainer_params)

model_params = {
    'vocab_words': trainer.vocab_words(),
    'embed_size': 100,
    'rnn_size': 64,
}
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='mdl',
    params=model_params)

estimator.train(input_fn=trainer.train_dataset, steps=500)
m = estimator.evaluate(input_fn=trainer.test_dataset)
print(m)

