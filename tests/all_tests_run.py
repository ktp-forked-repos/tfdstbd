from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .vocab import TestVocabulary
from .data import TestTrainer


if __name__ == "__main__":
    tf.test.main()
