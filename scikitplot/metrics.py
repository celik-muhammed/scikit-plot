"""
The :mod:`scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, average_precision_score,
    silhouette_score, silhouette_samples,
    auc, roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import (
    label_binarize, LabelEncoder
)
from sklearn.utils import deprecated
from sklearn.utils.multiclass import unique_labels
from sklearn.calibration import calibration_curve

from scikitplot.helpers import (
    validate_labels,
    binary_ks_curve, cumulative_gain_curve
)


def plot_calibration_curve(
    y_true, probas_list, clf_names=None, n_bins=10,
    title='Calibration plots (Reliability Curves)',
    ax=None, figsize=None, cmap='nipy_spectral',
    title_fontsize="large", text_fontsize="medium",
    pos_label=None, strategy="uniform",
):
    """Plots calibration curves for a set of classifier probability estimates.

    Plotting the calibration curves of a classifier is useful for determining
    whether or not you can interpret their predicted probabilities directly as
    as confidence level. For instance, a well-calibrated binary classifier
    should classify the samples such that for samples to which it gave a score
    of 0.8, around 80% should actually be from the positive class.

    This function currently only works for binary classification.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        probas_list (list of array-like, shape (n_samples, 2) or (n_samples,)):
            A list containing the outputs of binary classifiers'
            :func:`predict_proba` method or :func:`decision_function` method.

        clf_names (list of str, optional): A list of strings, where each string
            refers to the name of the classifier that produced the
            corresponding probability estimates in `probas_list`. If ``None``,
            the names "Classifier 1", "Classifier 2", etc. will be used.

        n_bins (int, optional): Number of bins. A bigger number requires more
            data.

        title (string, optional): Title of the generated plot. Defaults to
            "Calibration plots (Reliability Curves)".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

        pos_label (int, float, bool, str, optional): The positive label for binary
            classification. If `None`, the positive label is inferred from `y_true`.
            If `y_true` contains string labels or labels other than {0, 1} or {-1, 1},
            you must specify this parameter explicitly.

        strategy (str, optional): Strategy used to define the widths of the bins.
            uniform
                The bins have identical widths.
            quantile
                The bins have the same number of samples and depend on `y_prob`.

    Returns:
        :class:`matplotlib.axes.Axes`: The axes on which the plot was drawn.

    Example:
        >>> import scikitplot as skplt
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.metrics import calibration_curve
        >>> rf = RandomForestClassifier()
        >>> lr = LogisticRegression()
        >>> nb = GaussianNB()
        >>> svm = LinearSVC()
        >>> rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
        >>> lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
        >>> nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)
        >>> svm_scores = svm.fit(X_train, y_train).decision_function(X_test)
        >>> probas_list = [rf_probas, lr_probas, nb_probas, svm_scores]
        >>> clf_names = ['Random Forest', 'Logistic Regression',
        ...              'Gaussian Naive Bayes', 'Support Vector Machine']
        >>> skplt.metrics.plot_calibration_curve(y_test,
        ...                                      probas_list,
        ...                                      clf_names,
        ...                                      pos_label='1')
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_calibration_curve.png
           :align: center
           :alt: Calibration Curves
    """
    y_true = np.asarray(y_true)
    if not isinstance(probas_list, list):
        raise ValueError('`probas_list` does not contain a list.')

    classes = np.unique(y_true)
    if len(classes) > 2:
        raise ValueError('plot_calibration_curve only '
                         'works for binary classification')

    if clf_names is None:
        clf_names = ['Classifier {}'.format(x+1)
                     for x in range(len(probas_list))]

    if len(clf_names) != len(probas_list):
        raise ValueError('Length {} of `clf_names` does not match length {} of'
                         ' `probas_list`'.format(len(clf_names),
                                                 len(probas_list)))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i, probas in enumerate(probas_list):
        probas = np.asarray(probas)
        if probas.ndim > 2:
            raise ValueError('Index {} in probas_list has invalid '
                             'shape {}'.format(i, probas.shape))
        if probas.ndim == 2:
            probas = probas[:, 1]

        if probas.shape != y_true.shape:
            raise ValueError('Index {} in probas_list has invalid '
                             'shape {}'.format(i, probas.shape))

        probas = (probas - probas.min()) / (probas.max() - probas.min())

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, probas, n_bins=n_bins,
            pos_label=pos_label, strategy=strategy
        )
        color = plt.get_cmap(cmap)(float(i) / len(probas_list))

        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                label=clf_names[i], color=color)

    ax.set_title(title, fontsize=title_fontsize)

    ax.set_xlabel('Mean predicted value', fontsize=text_fontsize)
    ax.set_ylabel('Fraction of positives', fontsize=text_fontsize)

    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower right')

    return ax


def plot_confusion_matrix(
    y_true, y_pred, labels=None, true_labels=None,
    pred_labels=None, title=None, normalize=False,
    hide_zeros=False, hide_counts=False, x_tick_rotation=0, ax=None,
    figsize=None, cmap='Blues', title_fontsize="large",
    text_fontsize="medium", show_colorbar=True,
):
    """Generates confusion matrix plot from predictions and true labels

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_pred (array-like, shape (n_samples)):
            Estimated targets as returned by a classifier.

        labels (array-like, shape (n_classes), optional): List of labels to
            index the matrix. This may be used to reorder or select a subset
            of labels. If none is given, those that appear at least once in
            ``y_true`` or ``y_pred`` are used in sorted order. (new in v0.2.5)

        true_labels (array-like, optional): The true labels to display.
            If none is given, then all of the labels are used.

        pred_labels (array-like, optional): The predicted labels to display.
            If none is given, then all of the labels are used.

        title (string, optional): Title of the generated plot. Defaults to
            "Confusion Matrix" if `normalize` is True. Else, defaults to
            "Normalized Confusion Matrix.

        normalize (bool, optional): If True, normalizes the confusion matrix
            before plotting. Defaults to False.

        hide_zeros (bool, optional): If True, does not plot cells containing a
            value of zero. Defaults to False.

        hide_counts (bool, optional): If True, doe not overlay counts.
            Defaults to False.

        x_tick_rotation (int, optional): Rotates x-axis tick labels by the
            specified angle. This is useful in cases where there are numerous
            categories and the labels overlap each other.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

        show_colorbar (bool, optional): If False, does not add colour bar.
            Defaults to True.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> rf = RandomForestClassifier()
        >>> rf = rf.fit(X_train, y_train)
        >>> y_pred = rf.predict(X_test)
        >>> skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_confusion_matrix.png
           :align: center
           :alt: Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        validate_labels(classes, true_labels, "true_labels")

        true_label_indexes = np.isin(classes, true_labels)

        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        validate_labels(classes, pred_labels, "pred_labels")

        pred_label_indexes = np.isin(classes, pred_labels)

        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=title_fontsize)
    else:
        ax.set_title('Confusion Matrix', fontsize=title_fontsize)

    image = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))

    if show_colorbar == True:
        plt.colorbar(mappable=image)
        
    x_tick_marks = np.arange(len(pred_classes))
    y_tick_marks = np.arange(len(true_classes))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(pred_classes, fontsize=text_fontsize,
                       rotation=x_tick_rotation)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(true_classes, fontsize=text_fontsize)

    thresh = cm.max() / 2.

    if not hide_counts:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if not (hide_zeros and cm[i, j] == 0):
                ax.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=text_fontsize,
                        color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=text_fontsize)
    ax.set_xlabel('Predicted label', fontsize=text_fontsize)
    ax.grid(False)

    return ax


@deprecated(
    'This will be removed in v0.5.0. Please use '
    'scikitplot.metrics.plot_roc instead.'
)
def plot_roc_curve(y_true, y_probas, title='ROC Curves',
                   curves=('micro', 'macro', 'each_class'),
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "macro", "each_class")`
            i.e. "micro" for micro-averaged curve, "macro" for macro-averaged
            curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    if 'micro' not in curves and 'macro' not in curves and \
            'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro", "macro", or "each_class"')

    classes = np.unique(y_true)
    probas = y_probas

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, probas[:, i],
                                      pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(),
                                                  probas.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr[i], tpr[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(classes[i], roc_auc[i]))

    if 'micro' in curves:
        ax.plot(fpr[micro_key], tpr[micro_key],
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[micro_key]),
                color='deeppink', linestyle=':', linewidth=4)

    if 'macro' in curves:
        ax.plot(fpr[macro_key], tpr[macro_key],
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[macro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


def plot_roc(
    y_true,
    y_probas,
    title='ROC AUC Curves',
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap='nipy_spectral',
    class_index=1,
    multi_class=None,
    class_names=None,
    classes_to_plot=None,   
    plot_micro=True,
    plot_macro=True,
    show_labels=True,
    digits=3,
):
    """
    Generates the ROC AUC curves from labels and predicted scores/probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the `class_index`.

    title : str, optional, default='ROC AUC Curves'
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes on which to plot.
        If None, a new figure and axes are created.

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : str or matplotlib.colors.Colormap, optional, default='nipy_spectral'
        Colormap used for plotting.
        See Matplotlib Colormap documentation for options.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification.
        Ignored for binary classification.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot 
          for the entire probability distribution.

    class_names : list of str, optional, default=None
        List of class names for the legend.
        Order should match the order of classes in `y_probas`.

    classes_to_plot : list-like, optional, default=None
        Specific classes to plot. If given class does not exist,
        it will be ignored. If None, all classes are plotted.
        e.g. [0, 'cold']

    plot_micro : bool, optional, default=False
        Whether to plot the micro-average ROC AUC curve.

    plot_macro : bool, optional, default=False
        Whether to plot the macro-average ROC AUC curve.

    show_labels : bool, optional, default=True
        Whether to display the legend labels.

    digits : int, optional, default=3
        Number of digits for formatting AUC values in the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted ROC AUC curves.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> # from sklearn.datasets import load_iris as load_data  # multi
    >>> from sklearn.datasets import load_breast_cancer as load_data  # binary
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> import scikitplot as skplt
    >>> X, y = load_data(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> model = GaussianNB()
    >>> model.fit(X_train, y_train)
    >>> y_probas = model.predict_proba(X_test)
    >>> skplt.metrics.plot_roc(y_test, y_probas)
    <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
    >>> plt.show()

    Notes
    -----
    The implementation is specific to binary classification. For multiclass 
    problems, the 'ovr' or 'multinomial' strategies can be used. When 
    `multi_class='ovr'`, the plot focuses on the specified class (`class_index`).

    .. image:: _static/examples/plot_roc_curve.png
       :align: center
       :alt: ROC AUC Curves
    """
    title_pad = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)    

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # 1D y_probas (single class probabilities)
        if y_probas.ndim == 1:
            # Combine into a two-column
            y_probas = np.column_stack([1 - y_probas, y_probas])
    # Handle multi-class classification    
    elif len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = label_binarize(y_true, classes=np.unique(y_true))[:, class_index]
            # Handle 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                # Combine into a two-column binary format OvR
                y_probas = np.column_stack([1 - y_probas, y_probas])
            else:
                # Combine into a two-column binary format OvR
                y_probas = y_probas[:, class_index]
                y_probas = np.column_stack([1 - y_probas, y_probas])
                
            # Add a subtitle indicating the use of the One-vs-Rest strategy
            plt.suptitle(
                t="One-vs-Rest (OVR) strategy for multi-class classification.",
                fontsize=text_fontsize, x=0.512, y=0.902,
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='w', boxstyle='round,pad=0.2')
            )
            title_pad = 23
        elif multi_class in ['multinomial', None]:
            if y_probas.ndim == 1:
                raise ValueError(
                    "For multinomial classification, `y_probas` must be 2D."
                    "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                    "only 'ovr' multi-class strategy is supported."
                )
        else:
            raise ValueError("Unsupported `multi_class` strategy.")

    # Initialize dictionaries to store cumulative percentages and gains
    fpr_dict, tpr_dict = {}, {}

    # Get unique classes and filter those to be plotted
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate Curve for classes with only one category.'
        )
    classes_to_plot = classes if classes_to_plot is None else classes_to_plot
    indices_to_plot = np.isin(classes, classes_to_plot)
    
    # Binarize y_true for multiclass classification, for micro
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # Loop for all classes to get different class
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(
            y_true, y_probas[:, i], pos_label=classes[i]
        )
        roc_auc = auc(
            fpr_dict[i], tpr_dict[i]
        )        
        if to_plot:
            if class_names is None:
                class_names = classes            
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            ax.plot(
                fpr_dict[i], tpr_dict[i],
                ls='-', lw=2, color=color,
                label=(
                    f'Class {classes[i]} '
                    f'(area = {roc_auc:0>{digits}.{digits}f})'
                ),
            )

    # Whether or to plot macro or micro
    if plot_micro:
        fpr, tpr, _ = roc_curve(
            y_true_bin.ravel(), y_probas.ravel()
        )
        roc_auc = auc(
            fpr, tpr
        )
        ax.plot(
            fpr, tpr,
            ls=':', lw=3, color='deeppink',
            label=(
                'micro-average '
                f'(area = {roc_auc:0>{digits}.{digits}f})'
            ),
        )

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate(
            [ fpr_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += np.interp(
                all_fpr, fpr_dict[i], tpr_dict[i]
            )

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(
            all_fpr, mean_tpr
        )
        ax.plot(
            all_fpr, mean_tpr,
            ls=':', lw=3, color='navy',
            label=(
                'macro-average '
                f'(area = {roc_auc:0>{digits}.{digits}f})'
            ),
        )

    # Plot the baseline
    ax.plot([0, 1], [0, 1], ls='--', lw=1, c='gray', )  # label='Baseline'

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set x-axis ticks and labels
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:
        ax.legend(
            loc='lower right', 
            fontsize=text_fontsize, 
            title='ROC AUC', 
            alignment='left'
        )

    return ax


@deprecated(
    'This will be removed in v0.5.0. Please use '
    'scikitplot.metrics.plot_precision_recall instead.'
)
def plot_precision_recall_curve(
    y_true, y_probas,
    title='Precision-Recall Curve',
    curves=('micro', 'each_class'), ax=None,
    figsize=None, cmap='nipy_spectral',
    title_fontsize="large",
    text_fontsize="medium"
):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "each_class")`
            i.e. "micro" for micro-averaged curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if 'micro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro" or "each_class"')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true, probas[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i],
                                                       probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
        y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                           average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall[i], precision[i], lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(classes[i],
                                                     average_precision[i]),
                    color=color)

    if 'micro' in curves:
        ax.plot(recall[micro_key], precision[micro_key],
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision[micro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax


def plot_precision_recall(
    y_true,
    y_probas,
    title='Precision-Recall Curves',
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap='nipy_spectral',
    class_index=1,
    multi_class=None,
    class_names=None,
    classes_to_plot=None,   
    plot_micro=True,
    plot_macro=False,
    show_labels=True,
    digits=4,
    area='pr_auc',
):
    """
    Generates the Precision-Recall Curves from labels and predicted scores/probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the `class_index`.

    title : str, optional, default='Precision-Recall Curves'
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes on which to plot.
        If None, a new figure and axes are created.

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : str or matplotlib.colors.Colormap, optional, default='nipy_spectral'
        Colormap used for plotting.
        See Matplotlib Colormap documentation for options.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification.
        Ignored for binary classification.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot 
          for the entire probability distribution.

    class_names : list of str, optional, default=None
        List of class names for the legend.
        Order should match the order of classes in `y_probas`.

    classes_to_plot : list-like, optional, default=None
        Specific classes to plot. If given class does not exist,
        it will be ignored. If None, all classes are plotted.
        e.g. [0, 'cold']

    plot_micro : bool, optional, default=True
        Whether to plot the micro-average Precision-Recall AUC curve.

    plot_macro : bool, optional, default=False
        Whether to plot the macro-average Precision-Recall AUC curve.

    show_labels : bool, optional, default=True
        Whether to display the legend labels.

    digits : int, optional, default=3
        Number of digits for formatting AUC values in the plot.

    area : {'average_precision', 'pr_auc'}, optional, default='pr_auc'
        Strategy for calculating area score:
        - Both calculation gives very close results.
        - macro scores calculating by 'average_precision'.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted Precision-Recall AUC curves.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> # from sklearn.datasets import load_iris as load_data  # multi
    >>> from sklearn.datasets import load_breast_cancer as load_data  # binary
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> import scikitplot as skplt
    >>> X, y = load_data(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> model = GaussianNB()
    >>> model.fit(X_train, y_train)
    >>> y_probas = model.predict_proba(X_test)
    >>> skplt.metrics.plot_precision_recall(y_test, y_probas)
    <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
    >>> plt.show()

    Notes
    -----
    The implementation is specific to binary classification. For multiclass 
    problems, the 'ovr' or 'multinomial' strategies can be used. When 
    `multi_class='ovr'`, the plot focuses on the specified class (`class_index`).

    .. image:: _static/examples/plot_precision_recall_curve.png
       :align: center
       :alt: Precision-Recall AUC Curves
    """
    title_pad = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)    

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # 1D y_probas (single class probabilities)
        if y_probas.ndim == 1:
            # Combine into a two-column
            y_probas = np.column_stack([1 - y_probas, y_probas])
    # Handle multi-class classification    
    elif len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = label_binarize(y_true, classes=np.unique(y_true))[:, class_index]
            # Handle 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                # Combine into a two-column binary format OvR
                y_probas = np.column_stack([1 - y_probas, y_probas])
            else:
                # Combine into a two-column binary format OvR
                y_probas = y_probas[:, class_index]
                y_probas = np.column_stack([1 - y_probas, y_probas])
                
            # Add a subtitle indicating the use of the One-vs-Rest strategy
            plt.suptitle(
                t="One-vs-Rest (OVR) strategy for multi-class classification.",
                fontsize=text_fontsize, x=0.512, y=0.902,
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='w', boxstyle='round,pad=0.2')
            )
            title_pad = 23
        elif multi_class in ['multinomial', None]:
            if y_probas.ndim == 1:
                raise ValueError(
                    "For multinomial classification, `y_probas` must be 2D."
                    "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                    "only 'ovr' multi-class strategy is supported."
                )
        else:
            raise ValueError("Unsupported `multi_class` strategy.")

    # Initialize dictionaries to store cumulative percentages and gains
    precision_dict, recall_dict = {}, {}

    # Get unique classes and filter those to be plotted
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate Curve for classes with only one category.'
        )
    classes_to_plot = classes if classes_to_plot is None else classes_to_plot
    indices_to_plot = np.isin(classes, classes_to_plot)
    
    # Binarize y_true for multiclass classification, for micro
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # Loop for all classes to get different class
    for i, to_plot in enumerate(indices_to_plot):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            y_true, y_probas[:, i], pos_label=classes[i]
        )
        if area == 'pr_auc' :
            pr_auc = auc(
                recall_dict[i], precision_dict[i]
            )
        else:
            pr_auc = average_precision_score(
                y_true_bin[:, i], y_probas[:, i], #pos_label=classes[i]
            )
        if to_plot:
            if class_names is None:
                class_names = classes            
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            ax.plot(
                precision_dict[i], recall_dict[i],
                ls='-', lw=2, color=color,
                label=(
                    f'Class {classes[i]} '
                    f'(area = {pr_auc:0>{digits}.{digits}f})'
                ),
            )

    # Whether or to plot macro or micro
    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            y_true_bin.ravel(), y_probas.ravel()
        )
        if area == 'pr_auc' :
            pr_auc = auc(
                recall, precision
            )
        else:
            pr_auc = average_precision_score(
                y_true_bin.ravel(), y_probas.ravel()
            )
        ax.plot(
            precision, recall,
            ls=':', lw=3, color='deeppink',
            label=(
                'micro-average '
                f'(area = {pr_auc:0>{digits}.{digits}f})'
            ),
        )

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_precision = np.unique(np.concatenate(
            [ precision_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all ROC curves at this points
        mean_recall = np.zeros_like(all_precision)
        for i in range(len(classes)):
            mean_recall += np.interp(
                all_precision, precision_dict[i], recall_dict[i]
            )

        # Finally average it and compute AUC
        mean_recall /= len(classes)
        pr_auc = average_precision_score(
            y_true_bin.ravel(), y_probas.ravel(), average='macro'
        )
        ax.plot(
            all_precision, mean_recall,
            ls=':', lw=3, color='navy',
            label=(
                'macro-average '
                f'(area = {pr_auc:0>{digits}.{digits}f})'
            ),
        )

    # Plot the baseline
    ax.plot([0, 1], [1, 0], ls='--', lw=1, c='gray', )  # label='Baseline'

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('Recall', fontsize=text_fontsize)
    ax.set_ylabel('Precision', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set x-axis ticks and labels
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:
        ax.legend(
            loc='lower left', 
            fontsize=text_fontsize, 
            title=f'PR-AUC by {area}', 
            alignment='left'
        )

    return ax


def plot_silhouette(
    X, cluster_labels, title='Silhouette Analysis',
    metric='euclidean', copy=True, ax=None, figsize=None,
    cmap='nipy_spectral', title_fontsize="large",
    text_fontsize="medium", digits=3,
):
    """Plots silhouette analysis of clusters provided.

    Args:
        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.

        cluster_labels (array-like, shape (n_samples,)):
            Cluster label for each sample.

        title (string, optional): Title of the generated plot. Defaults to
            "Silhouette Analysis"

        metric (string or callable, optional): The metric to use when
            calculating distance between instances in a feature array.
            If metric is a string, it must be one of the options allowed by
            sklearn.metrics.pairwise.pairwise_distances. If X is
            the distance array itself, use "precomputed" as the metric.

        copy (boolean, optional): Determines whether ``fit`` is used on
            **clf** or on a copy of **clf**.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

        digits (int, optional): Number of digits for formatting output floating point values.
            Use e.g. 2 or 4. Defaults to 3.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> kmeans = KMeans(n_clusters=4, random_state=1)
        >>> cluster_labels = kmeans.fit_predict(X)
        >>> skplt.metrics.plot_silhouette(X, cluster_labels)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_silhouette.png
           :align: center
           :alt: Silhouette Plot
    """
    cluster_labels = np.asarray(cluster_labels)

    le = LabelEncoder()
    cluster_labels_encoded = le.fit_transform(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)

    sample_silhouette_values = silhouette_samples(X, cluster_labels,
                                                  metric=metric)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim([-0.1, 1])

    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10 + 10])

    ax.set_xlabel('Silhouette coefficient values', fontsize=text_fontsize)
    ax.set_ylabel('Cluster label', fontsize=text_fontsize)

    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels_encoded == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.get_cmap(cmap)(float(i) / n_clusters)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(le.classes_[i]),
                fontsize=text_fontsize)

        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg, color="red", linestyle="--",
        label='Silhouette score: {0:.{digits}f}'.format(silhouette_avg, digits=2)
    )

    ax.set_yticks([])  # Clear the y-axis labels / ticks
    ax.set_xticks(np.arange(-0.1, 1.0, 0.2))

    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)

    return ax


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    'plot_calibration_curve',
    'plot_confusion_matrix',
    'plot_roc_curve', 'plot_roc',
    'plot_precision_recall_curve', 'plot_precision_recall',
    'plot_silhouette',
]