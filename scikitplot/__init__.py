"""
This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
from . import (
    estimators,
    metrics,
    decomposition,
    cluster,
    deciles, 
)
from scikitplot.classifiers import classifier_factory
from scikitplot.clustering import clustering_factory

# __all__ = ['estimators']

# https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers
__version__ = '0.3.8dev13'