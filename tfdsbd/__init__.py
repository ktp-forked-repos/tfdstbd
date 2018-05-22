from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
import tensorflow as tf
import sysconfig
from os import path
from tensorflow.python.framework import ops


__VERSION__ = '0.1'


def __load_lib():
    uniq_flags = tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags() + [__VERSION__]
    uniq_flags = '/'.join(uniq_flags).encode('utf-8')
    flags_key = hashlib.md5(uniq_flags).hexdigest()

    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix is None:
        ext_suffix = sysconfig.get_config_var('SO')

    lib_file = 'tfdsbd_{}{}'.format(flags_key, ext_suffix)
    curr_dir = path.dirname(path.abspath(__file__))
    lib_path = path.join(curr_dir, '..', lib_file)

    if not path.exists(lib_path):
        raise Exception('OP library ({}) for your TF installation not found. '.format(lib_path) +
                        'Remove and install with "tfdsbd" package with --no-cache-dir option')

    return tf.load_op_library(lib_path)


_lib = __load_lib()


def features_length_case(source):
    """Extract length and case features from strings.

    Args:
        source: `Tensor` of any shape, strings to extract features.
    Returns:
        `Tensor` of source shape + 1 with float features.
    """

    source = tf.convert_to_tensor(source, dtype=tf.string)

    return _lib.features_length_case(source)


ops.NotDifferentiable("FeaturesLengthCase")



# from tfdsbd.tfdsbd.internal.model import model_fn



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
#         predictions = self.estimator._predict(input_fn=lambda: predict_input_fn([document]))
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
