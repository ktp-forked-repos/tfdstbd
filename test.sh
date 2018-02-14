#!/usr/bin/env bash

hash python 2>/dev/null && echo "Test with default Python" && python -m sentsegm.test.test
hash python2 2>/dev/null && echo "Test with Python v2" && python2 -m sentsegm.test.test
hash python3 2>/dev/null && echo "Test with Python v3" && python3 -m sentsegm.test.test
