from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(name='sentsegm',
      version='0.1',
      description='Sentence segmentation (boundary detection)',
      url='https://github.com/shkarupa-alex/sentsegm',
      author='Shkarupa Alex',
      author_email='shkarupa.alex@gmail.com',
      license='MIT',
      packages=['sentsegm'],
      install_requires=[
          'nltk',
          'tensorflow',
          'repoze.lru'
      ],
      zip_safe=False)
