from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from . import (
    estimators,
    decomposition,
    metrics, cluster,
    deciles, 
)
# https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers
__version__ = '0.3.8dev7'

from scikitplot.classifiers import classifier_factory
from scikitplot.clustering import clustering_factory