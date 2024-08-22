"""
An example showing the plot_silhouette method
used by a scikit-learn clusterer
"""
from sklearn.datasets import (
    make_classification,
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict
import numpy as np; np.random.seed(0)
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as skplt

# Load the data
X, y = data_3_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)

# Create an instance of the LogisticRegression
model = KMeans(n_clusters=4, random_state=0)

cluster_labels = model.fit_predict(X_train)

# Plot!
ax = skplt.metrics.plot_silhouette(
    X_train, cluster_labels
);
# Adjust layout to make sure everything fits
plt.tight_layout()
# Save the plot to a file
plt.savefig('plot_silhouette.png')
# Display the plot
plt.show(block=True)