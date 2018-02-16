# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .split_tokens import SplitTokensTest
from .extract_features import ExtractFeaturesTest

if __name__ == "__main__":
    tf.test.main()
