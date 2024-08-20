"""
The :mod:`scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve


def plot_feature_importances(
    clf,
    title='Feature Importances',
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap='PiYG',
    orientation='vertical',
    x_tick_rotation=None,
    bar_padding=11,
    display_labels=True,
    class_index=None,
    threshold=None,
    order=None,
    feature_names=None,
    digits=4,
):
    """
    Generates a plot of a classifier's feature importances.

    This function handles different types of classifiers and their respective
    feature importance or coefficient attributes. It supports models wrapped in pipelines.
    Error bars can be added based on different statistical methods or custom functions.

    Supports models like `LogisticRegression`, `RidgeClassifier`,
    `KNeighborsClassifier`, `LinearSVC`, `SVC`, `DecisionTreeClassifier`,
    `BaggingClassifier`, `RandomForestClassifier`, `Perceptron`,
    `BayesianRidge`, `HuberRegressor`, `TweedieRegressor`, 
    `LatentDirichletAllocation`, `PCA`, `LinearDiscriminantAnalysis`, and 
    `QuadraticDiscriminantAnalysis`.

    Parameters
    ----------
    clf : estimator object
        A fitted classifier or pipeline containing a classifier.

    title : str, optional
        Title of the generated plot.
        Defaults to "Feature Importances".

    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the curve. If None, the plot is drawn on
        a new set of axes.

    figsize : tuple, optional
        Tuple denoting figure size of the plot e.g. (6, 6). Defaults to None.

    title_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or
        integer-values. Defaults to "large".

    text_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or
        integer-values. Defaults to "medium".

    cmap : str or matplotlib.colors.Colormap, optional, default='PiYG'
        Colormap used for plotting.
        - See Matplotlib Colormap documentation for options.

    orientation : {'vertical', 'horizontal'}, optional
        Orientation of the bar plot. Defaults to 'vertical'.

    x_tick_rotation : int, optional
        Rotates x-axis tick labels by the specified angle. Defaults to None
        (automatically set based on orientation).

    bar_padding : float, optional
        Padding between bars in the plot. Defaults to 11.

    display_labels : bool, optional
        Whether to display the bar labels. Defaults to True.

    class_index : int, optional
        Index of the class of interest for multi-class classification.
        Defaults to None.

    threshold : float, optional
        Threshold for filtering features by absolute importance. Only
        features with an absolute importance greater than this threshold will
        be plotted. Defaults to None (plot all features).

    order : {'ascending', 'descending', None}, optional
        Order of feature importances in the plot. Defaults to None
        (automatically set based on orientation).

    feature_names : list of str, optional
        List of feature names corresponding to the features. If None, feature
        indices are used.

    digits : int, optional, default=4
        Number of digits for formatting AUC values in the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.
        
    feature_names : list of str
        List of feature names corresponding to the features. If None, feature
        indices are used.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # from sklearn.datasets import load_iris as load_data  # multi
    >>> from sklearn.datasets import load_breast_cancer as load_data  # binary
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.pipeline import make_pipeline
    >>> import scikitplot as skplt
    >>> X, y = load_data(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> clf = make_pipeline(StandardScaler(), RandomForestClassifier())
    >>> clf.fit(X_train, y_train)
    >>> skplt.estimators.plot_feature_importances(clf);
    <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
    >>> plt.show()
    """
    if figsize is None:
        figsize = (16, 9)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    # Handle pipelines
    if hasattr(clf, 'named_steps'):
        clf = clf.named_steps[next(reversed(clf.named_steps))]

    # Determine the appropriate attribute for feature importances or coefficients
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        if clf.coef_.ndim > 1:  # Multi-class case
            if class_index is not None:
                importances = np.array(clf.coef_)[class_index]
            else:
                importances = np.mean(np.abs(clf.coef_), axis=0)
        else:
            importances = clf.coef_.ravel()
    elif hasattr(clf, 'explained_variance_ratio_'):  # PCA, LDA
        importances = clf.explained_variance_ratio_
    elif hasattr(clf, 'lda_'):  # LDA (scikit-learn < 0.24)
        importances = np.abs(clf.lda_.scalings_).ravel()
    elif hasattr(clf, 'coef_'):  # QDA (scikit-learn >= 0.24)
        importances = np.abs(clf.coef_).ravel()
    else:
        raise TypeError(
            'The estimator does not have an attribute for feature '
            'importances or coefficients.'
        )
    # Obtain feature names
    if feature_names is None:
        if hasattr(clf, 'feature_names_in_'):
            feature_names = clf.feature_names_in_
        else:
            feature_names = np.arange(len(importances))
    else:
        feature_names = np.array(feature_names)

    # Apply filtering based on the threshold
    indices = np.arange(len(importances))

    if threshold is not None:
        mask = np.abs(importances) > threshold
        importances = importances[mask]
        feature_names = feature_names[mask]        
        indices = np.arange(len(importances))
    
    # Apply ordering based on orientation
    if order is None:
        order = 'ascending' if orientation == 'horizontal' else 'descending'
    if order == 'descending':
        indices = indices[np.argsort(importances)[::-1]]
    elif order == 'ascending':
        indices = indices[np.argsort(importances)]

    importances = importances[indices]
    feature_names = feature_names[indices]
    
    # Prepare the color map
    cmap_obj = plt.get_cmap(cmap)
    
    # Plot bars based on orientation
    for idx, imp in enumerate(importances):
        color = cmap_obj(float(idx) / len(importances))
        
        if orientation == 'vertical':
            bar = ax.bar(x=feature_names[idx], height=imp, color=color)
        elif orientation == 'horizontal':
            bar = ax.barh(y=feature_names[idx], width=imp, color=color)
        else:
            raise ValueError(
                "Invalid value for orientation: "
                "must be 'vertical' or 'horizontal'."
            )

    # Set default x_tick_rotation based on orientation
    if x_tick_rotation is None:
        x_tick_rotation = 0 if orientation == 'horizontal' else 90

    if display_labels:
        for bars in ax.containers:
            ax.bar_label(
                bars,
                fmt=lambda x:'{:0>{digits}.{digits}f}'.format(x, digits=digits),
                fontsize=text_fontsize, 
                rotation=x_tick_rotation,
                padding=bar_padding,
            )
    ax.set_title(title, fontsize=title_fontsize)
    if orientation == 'vertical':
        ax.tick_params(axis='x', rotation=x_tick_rotation)
        # ax.set_xticks(np.arange(len(importances)))
        # ax.set_xticklabels(feature_names, rotation=x_tick_rotation)
        ax.set_xlabel("Features", fontsize=text_fontsize)
        ax.set_ylabel("Importance", fontsize=text_fontsize)
    elif orientation == 'horizontal':
        ax.tick_params(axis='y', rotation=x_tick_rotation)
        # ax.set_yticks(np.arange(len(importances)))
        # ax.set_yticklabels(feature_names, rotation=x_tick_rotation)
        ax.set_xlabel("Importance", fontsize=text_fontsize)
        ax.set_ylabel("Features", fontsize=text_fontsize)
    else:
        raise ValueError(
            "Invalid value for orientation: must be "
            "'vertical' or 'horizontal'."
        )
    plt.tight_layout()
    plt.legend([f'features: {len(importances)}'])
    return ax, feature_names


def plot_learning_curve(
    clf, X, y, title='Learning Curve',
    cv=None, shuffle=False, random_state=None,
    train_sizes=None, n_jobs=1, scoring=None,
    ax=None, figsize=None, title_fontsize="large",
    text_fontsize="medium",
):
    """Generates a plot of the train and test learning curves for a classifier.

    Args:
        clf: Classifier instance that implements ``fit`` and ``predict``
            methods.

        X (array-like, shape (n_samples, n_features)):
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for classification or regression;
            None for unsupervised learning.

        title (string, optional): Title of the generated plot. Defaults to
            "Learning Curve"

        cv (int, cross-validation generator, iterable, optional): Determines
            the cross-validation strategy to be used for splitting.

            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is
            used.

        shuffle (bool, optional): Used when do_cv is set to True. Determines
            whether to shuffle the training data before splitting using
            cross-validation. Default set to True.

        random_state (int :class:`RandomState`): Pseudo-random number generator
            state used for random sampling.

        train_sizes (iterable, optional): Determines the training sizes used to
            plot the learning curve. If None, ``np.linspace(.1, 1.0, 5)`` is
            used.

        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            1.

        scoring (string, callable or None, optional): default: None
            A string (see scikit-learn model evaluation documentation) or a
            scorerbcallable object / function with signature
            scorer(estimator, X, y).

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

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
        >>> rf = RandomForestClassifier()
        >>> skplt.estimators.plot_learning_curve(rf, X, y)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_learning_curve.png
           :align: center
           :alt: Learning Curve
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Training examples", fontsize=text_fontsize)
    ax.set_ylabel("Score", fontsize=text_fontsize)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring, shuffle=shuffle, random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="best", fontsize=text_fontsize)

    return ax


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    'plot_feature_importances',
    'plot_learning_curve',
]