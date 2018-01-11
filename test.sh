#!/usr/bin/env bash

hash python 2>/dev/null && python -m sentsegm.test
hash python2 2>/dev/null && python2 -m sentsegm.test
hash python3 2>/dev/null && python3 -m sentsegm.test
