from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.estimator import multi_head
from tfseqestimator import SequenceItemsClassifier
from tfseqestimator.head import sequence_binary_classification_head_with_sigmoid


class SentenceTokenEstimator(SequenceItemsClassifier):
    def _estimator_head(self, weight_column):
        token_head = sequence_binary_classification_head_with_sigmoid(
            weight_column=weight_column,
            label_vocabulary=self.label_vocabulary,  # Both heads should have same label vocabulary
            loss_reduction=self.loss_reduction,
            name='tokens'
        )
        sentence_head = sequence_binary_classification_head_with_sigmoid(
            weight_column=weight_column,
            label_vocabulary=self.label_vocabulary,  # Both heads should have same label vocabulary
            loss_reduction=self.loss_reduction,
            name='sentences'
        )

        return multi_head([token_head, sentence_head])
