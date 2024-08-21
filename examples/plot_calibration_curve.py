"""
An example showing the plot_calibration_curve method
used by a scikit-learn classifier
"""
from sklearn.datasets import (
    make_classification,
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np; np.random.seed(0)
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as skplt

# Load the data
X, y = make_classification(
    n_samples=100000, 
    n_features=20,
    n_informative=2, 
    n_redundant=2,
    random_state=0
)
X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]

# Create an instance of the LogisticRegression
lr_probas = LogisticRegression().fit(X_train, y_train).predict_proba(X_val)
nb_probas = GaussianNB().fit(X_train, y_train).predict_proba(X_val)
svc_scores = LinearSVC().fit(X_train, y_train).decision_function(X_val)
rf_probas = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_val)

probas_dict = {
    'Logistic Regression': lr_probas,
    'Gaussian Naive Bayes': nb_probas,
    'Support Vector Classifier': svc_scores,
    'Random Forest Classifier': rf_probas,
}
# Plot!
ax = skplt.metrics.plot_calibration_curve(
    y_val,
    clf_names=list(probas_dict.keys()),
    probas_list=list(probas_dict.values()),
    n_bins=10
);
# Adjust layout to make sure everything fits
plt.tight_layout()
# Save the plot to a file
plt.savefig('plot_calibration_curve.png')
# Display the plot
plt.show(block=True)