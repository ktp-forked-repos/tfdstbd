from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from six import string_types
from .input import predict_input_fn
from .model import model_fn
from .param import model_params
from .train import Trainer


# class SentenceBoundaryDetector:
#     def __init__(self, model_dir):
#         model_dir = os.path.abspath(model_dir)
#         assert os.path.exists(model_dir), 'model_dir {} does not exist'.format(model_dir)
#         assert os.path.isdir(model_dir), 'model_dir {} is not a directory'.format(model_dir)
#
#         trainer_params = {
#             'data_dir': os.path.join(os.path.dirname(__file__), '..', 'sentsegm', 'data'),
#             'batch_size': 100,
#             'test_size': 0.2,
#             'doc_size': 100,
#         }
#         trainer = Trainer(**trainer_params)
#
#         params = model_params()
#         params['vocab_words'] = trainer.vocab_words()
#         params['class_weights'] = trainer.train_weights()
#
#         self.estimator = tf.estimator.Estimator(
#             model_fn=model_fn,
#             model_dir=model_dir,
#             params={} #params # TODO check if required
#         )
#
#     def split(self, document):
#         assert isinstance(document, string_types)
#
#         predictions = self.estimator.predict(input_fn=lambda: predict_input_fn([document]))
#         assert 'document' in predictions
#         assert 'class' in predictions
#         assert len(predictions['words']) == len(predictions['classes']) == 1
#
#         words = predictions['document'][0]
#         classes = predictions['class'][0]
#
#         sentences = []
#         sentence = []
#         for word, boundary in zip(words, classes):
#             sentence.append(word)
#             if boundary:
#                 sentences.append(u''.join(sentence))
#                 sentence.clear()
#
#         return sentences
