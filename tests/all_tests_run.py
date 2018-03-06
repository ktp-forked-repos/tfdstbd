from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .convert import TestTokenizeDataset, TestMakeDataset, TestWriteDataset

if __name__ == "__main__":
    tf.test.main()
