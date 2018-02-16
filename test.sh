#!/usr/bin/env bash

hash python 2>/dev/null && echo "Test with default Python" && python -m sentsegm.test.all
hash python2 2>/dev/null && echo "Test with Python v2" && python2 -m sentsegm.test.all
hash python3 2>/dev/null && echo "Test with Python v3" && python3 -m sentsegm.test.all
