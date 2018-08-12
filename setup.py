from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tfdsbd',
    version='1.0.0',
    description='Deep sentence boundary detection implemented with Tensorflow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/shkarupa-alex/tfdsbd',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=1.9.0',
        'tfseqestimator>=1.1.0',
        'tfscc3d>=1.0.0',
        'tfunicode>=1.4.4',
        'nlpvocab>=1.0.0',
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)
