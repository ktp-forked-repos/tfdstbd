#!/usr/bin/env bash

hash python 2>/dev/null && python -m unittest test
hash python2 2>/dev/null && python2 -m unittest test
hash python3 2>/dev/null && python3 -m unittest test
