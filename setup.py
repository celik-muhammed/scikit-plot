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


## Basic information
HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text(encoding='utf-8')
VERSION = get_version('scikitplot/__init__.py')

## Define the keywords
KEYWORDS = [
    'visualization',
    'machine learning',
    'scikit-learn',
    'matplotlib',
    'data science',
]

##########################################################################
## Define the configuration, Run setup script
##########################################################################

setup(
    name='scikit-plots',
    version=VERSION,
    description='An intuitive library to add plotting functionality to scikit-learn objects.',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    license='MIT License',
    author='Reiichiro Nakano et al.',  # Your name first as you're the current maintainer
    author_email='reiichiro.s.nakano@gmail.com',  # Your email address
    url='https://github.com/celik-muhammed/scikit-plot/tree/muhammed-dev',  # Your fork's URL
    maintainer='Muhammed Çelik',
    maintainer_email='muhammed.business.network@gmail.com',
    project_urls={
        'Documentation': 'https://scikit-plot.readthedocs.io/en/stable/',
        'Source Code': 'https://github.com/celik-muhammed/scikit-plot',  # Updated to your fork's URL
        'Bug Tracker': 'https://github.com/celik-muhammed/scikit-plot/issues',  # Updated to your fork's issues URL
        'Forum': 'https://github.com/celik-muhammed/scikit-plot/issues',  # Updated forum link
        'Donate': 'https://github.com/celik-muhammed/scikit-plot#donate',  # Updated donation link
    },
    download_url='https://github.com/celik-muhammed/scikit-plot/tree/muhammed-dev',  # Your fork's download URL
    # packages=find_packages(),  # Finds all packages automatically
    packages=['scikitplot'],
    include_package_data=True,
    platforms='any',
    # entry_points={"console_scripts": []},
    install_requires=[
        'matplotlib>=1.4.0',
        'scikit-learn>=0.21',
        'scipy>=0.9',
        'joblib>=0.10'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',  # Change status as per the current state
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    cmdclass={'test': PyTest},
    tests_require=['pytest'],
    test_suite='scikitplot.tests.test_scikitplot',
    extras_require={
        'testing': ['pytest'],
    }
)