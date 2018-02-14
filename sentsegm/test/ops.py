# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .split_tokens import SplitTokensTest
from .extract_features import ExtractFeaturesTest
import tensorflow as tf


if __name__ == "__main__":
    tf.test.main()
