from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import tensorflow as tf
# from six import string_types
# from .dataset import predict_input_fn
# from .model import model_fn
# from .params import model_params
#
#
# class SentenceBoundaryDetector:
#     def __init__(self, model_dir):
#         model_dir = os.path.abspath(model_dir)
#         assert os.path.exists(model_dir), 'model_dir {} does not exist'.format(model_dir)
#         assert os.path.isdir(model_dir), 'model_dir {} is not a directory'.format(model_dir)
#
#         self.estimator = tf.estimator.Estimator(
#             model_fn=model_fn,
#             model_dir=model_dir,
#             params=params # TODO check if required
#         )
#
#     def split(self, document):
#         assert isinstance(document, string_types)
#
#         predictions = self.estimator.predict(input_fn=predict_input_fn([document]))
#         assert 'words' in predictions
#         assert 'classes' in predictions
#         assert len(predictions['words']) == len(predictions['classes']) == 1
#
#         words = predictions['words'][0]
#         classes = predictions['classes'][0]
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
