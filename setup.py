"""
Setup script.
"""
from __future__ import print_function

import os
import io
import sys
import codecs
import pathlib

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

def read(*filenames, **kwargs):
    here = os.path.abspath(os.path.dirname(__file__))
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(os.path.join(here, filename), encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError('Unable to find version string.')


# HERE = os.path.abspath(os.path.dirname(__file__))
HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()
VERSION = get_version('scikitplot/__init__.py')

setup(
    name='scikit-plot',
    description='An intuitive library to add plotting functionality to scikit-learn objects.',
    long_description_content_type="text/markdown",
    long_description=README,
    version=VERSION,
    url='https://github.com/reiinakano/scikit-plot',
    author='Reiichiro Nakano',
    author_email='reiichiro.s.nakano@gmail.com',
    license='MIT License',
    install_requires=[
        'matplotlib>=1.4.0',
        'scikit-learn>=0.18',
        'scipy>=0.9',
        'joblib>=0.10'
    ],
    # Supported Python versions
    # python_requires=">=2.7",
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        ],
    packages=['scikitplot'],
    include_package_data=True,
    platforms='any',
    tests_require=['pytest'],
    test_suite='scikitplot.tests.test_scikitplot',
    cmdclass={'test': PyTest},
    extras_require={
        'testing': ['pytest'],
    }
)