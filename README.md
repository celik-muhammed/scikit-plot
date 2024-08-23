<h1 style="text-align:center;">Welcome to 101 Scikit-plots</h1>

<h3 style="text-align:left;">Single line functions for detailed visualizations</h3>
<h3 style="text-align:left;">The quickest and easiest way to go from analysis...</h3>

<table style="margin-left:auto;margin-right:auto;width:100%;border-collapse:collapse;">
  <tr>
    <th style="width:50%;text-align:center;">Sample Plots</th>
    <th style="width:50%;text-align:center;">Sample Plots</th>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_learning_curve.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_learning_curve.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_calibration_curve.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_calibration_curve.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_classifier_eval.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_classifier_eval.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_feature_importances.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_feature_importances.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_roc.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_roc.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_precision_recall.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_precision_recall.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_pca_component_variance.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_pca_component_variance.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_pca_2d_projection.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_pca_2d_projection.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_elbow.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_elbow.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_silhouette.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_silhouette.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_cumulative_gain.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_cumulative_gain.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_lift.png" src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_lift.png">
    </td>
  </tr>
</table>


<p>
  Scikit-plot is the result of an unartistic data scientist's dreadful realization that 
  <em>visualization is one of the most crucial components in the data science process, not just a mere afterthought</em>.
</p>

<p>
  Gaining insights is simply a lot easier when you're looking at a colored heatmap of a confusion matrix complete with 
  class labels rather than a single-line dump of numbers enclosed in brackets. Besides, if you ever need to present your results to 
  someone (virtually any time anybody hires you to do data science), you show them visualizations, not a bunch of numbers in Excel.
</p>

<p>
  That said, there are a number of visualizations that frequently pop up in machine learning. Scikit-plot is a humble attempt to provide 
  aesthetically-challenged programmers (such as myself) the opportunity to generate quick and beautiful graphs and plots with as little 
  boilerplate as possible.
</p>

<h2 style="text-align:left;">Okay then, prove it. Show us an example.</h2>

<p>
  Say we use Naive Bayes in multi-class classification and decide we want to visualize the results of a common classification metric, 
  the Area under the Receiver Operating Characteristic curve. Since the ROC is only valid in binary classification, we want to show 
  the respective ROC of each class if it were the positive class. As an added bonus, let's show the micro-averaged and macro-averaged 
  curve in the plot as well.
</p>

<p>
  Let's use scikit-plot with the sample digits dataset from scikit-learn.
</p>

<pre>
<code class="language-python">
# The usual train-test split mumbo-jumbo
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

# Load the data
X, y = data_10_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)

# Create an instance of the LogisticRegression
model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)

# Perform predictions
y_val_prob = model.predict_proba(X_val)

# The magic happens here
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas);
</code>
</pre>

<p style="text-align:center;">
  <img src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/examples/plot_roc.png" alt="roc_curves" style="width:80%;">
</p>

<p style="text-align:center;">
  Pretty.
</p>

<p>
  And... That's it. Encaptured in that small example is the entire philosophy of Scikit-plot: 
  <strong>single line functions for detailed visualization</strong>. You simply browse the plots available in the documentation, 
  and call the function with the necessary arguments. Scikit-plot tries to stay out of your way as much as possible. 
  No unnecessary bells and whistles. And when you <em>do</em> need the bells and whistles, each function offers a myriad of 
  parameters for customizing various elements in your plots.
</p>

<p>
  Finally, compare and <a href="http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html">view the non-scikit-plot way of plotting the multi-class ROC curve</a>. 
  Which one would you rather do?
</p>

<h2 style="text-align:center;">Maximum flexibility. Compatibility with non-scikit-learn objects.</h2>

<p>
  Although Scikit-plot is loosely based around the scikit-learn interface, you don't actually need Scikit-learn objects to use the available functions. 
  As long as you provide the functions what they're asking for, they'll happily draw the plots for you.
</p>

<p>
  Here's a quick example to generate the precision-recall curves of a Keras classifier on a sample dataset.
</p>

<pre>
<code class="language-python">
# Import what's needed for the Functions API
import matplotlib.pyplot as plt
import scikitplot as skplt

# This is a Keras classifier. We'll generate probabilities on the test set.
keras_clf.fit(X_train, y_train, batch_size=64, nb_epoch=10, verbose=2)
probas = keras_clf.predict_proba(X_test, batch_size=64)

# Now plot.
skplt.metrics.plot_precision_recall_curve(y_test, probas);
</code>
</pre>

<p style="text-align:center;">
  <img src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/master/examples/p_r_curves.png" alt="p_r_curves" style="width:80%;">
</p>

<p>
  You can see clearly here that <code>skplt.metrics.plot_precision_recall_curve</code> needs only the ground truth y-values and the predicted probabilities 
  to generate the plot. This lets you use <em>anything</em> you want as the classifier, from Keras NNs to NLTK Naive Bayes to that groundbreaking classifier 
  algorithm you just wrote.
</p>

<p>
  The possibilities are endless.
</p>

<h2 style="text-align:center;">Plot Decile Table, Lift, Gain and KS Statistic charts with single line functions</h2>

<blockquote>
  Migrating as <code>skplt.deciles</code> module: <a href="https://github.com/tensorbored/kds">https://github.com/tensorbored/kds</a>
</blockquote>

<pre>
<code>
# Import what's needed for the Functions API
import matplotlib.pyplot as plt
import scikitplot as skplt

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_prob = clf.predict_proba(X_val)

# Now plot.
skplt.deciles.report(y_val, y_prob[:,1], plot_style='ggplot')
</code>
</pre>

<p>
  Choose among multiple <code>plot_style</code> list using <code>plt.style.available</code>, to generate quick and beautiful plots.
</p>

<p style="text-align:center;">
  <img src="https://raw.githubusercontent.com/celik-muhammed/scikit-plot/muhammed-dev/docs/_static/readme_report.png" alt="readme_report.png" style="width:100%;">
</p>


## Installation

Installation is simple! First, make sure you have the dependencies [Scikit-learn](http://scikit-learn.org) and [Matplotlib](http://matplotlib.org/) installed.

Then just run:
```bash
pip install scikit-plots
```

Or if you want the latest development version, clone this repo and run
```bash
python setup.py install
```
at the root folder.

<!-- If using conda, you can install Scikit-plot by running:
```bash
conda install -c conda-forge scikit-plot
``` -->

## Documentation and Examples

Explore the full features of Scikit-plot.

You can find detailed documentation [here](http://scikit-plot.readthedocs.io).

Examples are found in the [examples folder of this repo](examples/).

## Contributing to Scikit-plots

Reporting a bug? Suggesting a feature? Want to add your own plot to the library? Visit our [contributor guidelines](CONTRIBUTING.md).

## Citing Scikit-plots

Are you using Scikit-plots in an academic paper? You should be! Reviewers love eye candy.

If so, please consider citing Scikit-plots with:
- Genereal DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13367001.svg)](https://doi.org/10.5281/zenodo.13367001)


#### IEEE

> [1]scikit-plots, “scikit-plots: v0.3.8dev0”. Zenodo, Ağu. 23, 2024. doi: 10.5281/zenodo.13367001.

#### APA

> scikit-plots. (2024). scikit-plots: v0.3.8dev0 (0.3.8dev0). Zenodo. https://doi.org/10.5281/zenodo.13367001

#### Harvard

> scikit-plots (2024) “scikit-plots: v0.3.8dev0”. Zenodo. doi: 10.5281/zenodo.13367001.

Happy plotting!