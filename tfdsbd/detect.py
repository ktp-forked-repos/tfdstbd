# -*- coding: utf-8 -*-
import os
import re
import sys

from tfunicode import transform_normalize_unicode,transform_lower_case, transform_zero_digits, \
    transform_wrap_with, transform_upper_case,  expand_split_words, expand_char_ngrams, expand_split_chars
from tensorflow.contrib.saved_model import get_signature_def_by_key
from tensorflow.python.saved_model import loader
from tensorflow.python.tools import saved_model_utils
import tensorflow as tf


class SentenceBoundaryDetector:
    def __init__(self, model_dir):
        model_dir = os.path.abspath(model_dir)
        assert os.path.exists(model_dir), 'model_dir {} does not exist'.format(model_dir)
        assert os.path.isdir(model_dir), 'model_dir {} is not a directory'.format(model_dir)

        meta_graph_def = saved_model_utils.get_meta_graph_def(model_dir, 'serve')
        self.inputs_tensor_info = get_signature_def_by_key(meta_graph_def, 'predict').inputs
        outputs_tensor_info = get_signature_def_by_key(meta_graph_def, 'predict').outputs
        # Sort to preserve order because we need to go from value to key later.
        self.output_tensor_keys = sorted(outputs_tensor_info.keys())
        self.output_tensor_names = [outputs_tensor_info[tensor_key].name for tensor_key in self.output_tensor_keys]

        self.session = tf.Session(graph=tf.Graph())
        loader.load(self.session, ['serve'], model_dir)

    def __del__(self):
        self.session.close()

    def _predict(self, documents):
        output_tensor_values = self.session.run(
            self.output_tensor_names,
            feed_dict={
                self.inputs_tensor_info['input'].name: documents
            }
        )

        result = {}
        for i, value in enumerate(output_tensor_values):
            key = self.output_tensor_keys[i]
            result[key] = value

        return result

    def detect(self, documents):
        assert isinstance(documents, list)

        predictions = self._predict(documents)
        assert 'tokens' in predictions
        assert 'classes' in predictions

        results = []
        for tokens, classes in zip(predictions['tokens'], predictions['classes']):
            assert len(tokens) == len(classes)

            sentences = []
            buffer = []
            for t, c in zip(tokens, classes):
                buffer.append(t)
                if c:
                    sentences.append(b''.join(buffer).decode('utf-8'))
                    buffer.clear()
            sentences = [re.sub('\s+', ' ', s).strip() for s in sentences]
            sentences = [s for s in sentences if len(s)]

            results.append(sentences)

        return results
