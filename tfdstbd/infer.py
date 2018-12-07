from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import tensorflow as tf
import tfunicode  # required to load custom ops
from tensorflow.contrib.saved_model import get_signature_def_by_key
from tensorflow.python.saved_model import loader
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.estimator.canned import prediction_keys


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

    def split(self, documents):
        assert isinstance(documents, list)

        predictions = self._predict(documents)
        classes_key = prediction_keys.PredictionKeys.CLASSES
        assert 'words' in predictions
        assert classes_key in predictions

        results = []
        for tokens, classes in zip(predictions['words'], predictions[classes_key]):
            assert len(tokens) == len(classes)

            sentences = []
            words = []
            for t, c in zip(tokens, classes):
                if b'' == t:
                    continue

                words.append(t)

                if b'B' == c:
                    sentences.append(b''.join(words).decode('utf-8'))
                    words = []

            sentences.append(b''.join(words).decode('utf-8'))

            sentences = [re.sub('\s+', ' ', s.strip()) for s in sentences]
            sentences = [s for s in sentences if len(s)]

            results.append(sentences)

        return results


def main():
    parser = argparse.ArgumentParser(description='Split text into tokens')
    parser.add_argument(
        'model_dir',
        type=str,
        help='Exported tfdstbd model directory')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help='Input text file')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.model_dir) and os.path.isdir(argv.model_dir)

    tf.logging.set_verbosity(tf.logging.INFO)

    document = argv.src_file.read().decode('utf-8')
    sbd = SentenceBoundaryDetector(argv.model_dir)
    tokens = sbd.split([document])
    separator = '\n\n{}\n\n'.format('-' * 80)
    print(separator.join(tokens[0]))
