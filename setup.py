from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(name='tfdsbd',
      version='0.1',
      description='Deep sentence boundary detection implemented with Tensorflow',
      url='https://github.com/shkarupa-alex/tfdsbd',
      author='Shkarupa Alex',
      author_email='shkarupa.alex@gmail.com',
      license='MIT',
      packages=['tfdsbd'],
      install_requires=[
          'tensorflow>=1.5.0',
      ])
