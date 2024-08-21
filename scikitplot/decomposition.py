"""
This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.

The :mod:`scikitplot.decomposition` module includes plots built specifically
for scikit-learn estimators that are used for dimensionality reduction
e.g. PCA. You can use your own estimators, but these plots assume specific
properties shared by scikit-learn estimators. The specific requirements are
documented per function.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    'plot_pca_component_variance',
    'plot_pca_2d_projection',
]


def plot_pca_component_variance(
    clf, 
    title='PCA Component Explained Variances',
    ax=None, 
    figsize=None, 
    title_fontsize="large",
    text_fontsize="medium",
    target_explained_variance=0.75,
    x_tick_rotation=90,
):
    """Plots PCA components' explained variance ratios. (new in v0.2.2)

    Args:
        clf: PCA instance that has the ``explained_variance_ratio_`` attribute.

        title (string, optional): Title of the generated plot. Defaults to
            "PCA Component Explained Variances"

        target_explained_variance (float, optional): Looks for the minimum
            number of principal components that satisfies this value and
            emphasizes it on the plot. Defaults to 0.75

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
            
        x_tick_rotation : int, optional
            Rotates x-axis tick labels by the specified angle. Defaults to None
            (automatically set based on orientation).

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> pca = PCA(random_state=1)
        >>> pca.fit(X)
        >>> skplt.decomposition.plot_pca_component_variance(pca)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_pca_component_variance.png
           :align: center
           :alt: PCA Component variances
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    if not hasattr(clf, 'explained_variance_ratio_'):
        raise TypeError('"clf" does not have explained_variance_ratio_ '
                        'attribute. Has the PCA been fitted?')

    cumulative_sum_ratios = np.cumsum(clf.explained_variance_ratio_)

    # Magic code for figuring out closest value to target_explained_variance
    idx = np.searchsorted(cumulative_sum_ratios, target_explained_variance)

    ax.plot(range(len(clf.explained_variance_ratio_) + 1),
            np.concatenate(([0], np.cumsum(clf.explained_variance_ratio_))),
            '*-')
    
    if idx < len(cumulative_sum_ratios):
        ax.plot(idx+1, cumulative_sum_ratios[idx], 'ro',
                label='{0:0.3f} Explained variance ratio for '
                'first {1} components'.format(cumulative_sum_ratios[idx],
                                              idx+1),
                markersize=4, markeredgewidth=4)
        ax.axhline(cumulative_sum_ratios[idx],
                   linestyle=':', lw=3, color='black')

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(
        'First n principal components', 
        fontsize=text_fontsize
    )
    ax.set_ylabel(
        'Explained variance ratio of first n components',
        fontsize=text_fontsize
    )
    ax.tick_params(labelsize=text_fontsize)
    ax.tick_params(axis='x', rotation=x_tick_rotation)
    
    # ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.02, 1.02])
    
    # Set x-axis ticks and labels
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    
    # Enable grid and display legend
    ax.grid(True)
    ax.legend(
        loc="best",
        fontsize=text_fontsize
    )
    plt.tight_layout()
    return ax


def plot_pca_2d_projection(
    clf, 
    X, 
    y, 
    title='PCA 2-D Projection',
    ax=None, 
    figsize=None, 
    title_fontsize="large",
    text_fontsize="medium", 
    cmap='Spectral',
    biplot=False, 
    feature_labels=None,
    dimensions=[0, 1], 
    label_dots=False, 
):
    """Plots the 2-dimensional projection of PCA on a given dataset.

    Args:
        clf: Fitted PCA instance that can ``transform`` given data set into 2
            dimensions.

        X (array-like, shape (n_samples, n_features)):
            Feature set to project, where n_samples is the number of samples
            and n_features is the number of features.

        y (array-like, shape (n_samples) or (n_samples, n_features)):
            Target relative to X for labeling.

        title (string, optional): Title of the generated plot. Defaults to
            "PCA 2-D Projection"

        biplot (bool, optional): If True, the function will generate and plot
        	biplots. If false, the biplots are not generated.

        feature_labels (array-like, shape (n_classes), optional): List of labels
        	that represent each feature of X. Its index position must also be
        	relative to the features. If ``None`` is given, then labels will be
        	automatically generated for each feature.
        	e.g. "variable1", "variable2", "variable3" ...

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

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> pca = PCA(random_state=1)
        >>> pca.fit(X)
        >>> skplt.decomposition.plot_pca_2d_projection(pca, X, y)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_pca_2d_projection.png
           :align: center
           :alt: PCA 2D Projection
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    transformed_X = clf.transform(X)
    # Get unique classes from y, preserving order of class occurence in y
    _, class_indexes = np.unique(np.array(y), return_index=True)
    classes = np.array(y)[np.sort(class_indexes)]

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(classes)))
    for label, color in zip(classes, colors):
        ax.scatter(transformed_X[y == label, dimensions[0]], transformed_X[y == label, dimensions[1]],
                   alpha=0.8, lw=2, label=label, color=color)

        if label_dots:
            for dot in transformed_X[y == label][:, dimensions]:
                ax.text(*dot, label)

    if biplot:
        xs = transformed_X[:, dimensions[0]]
        ys = transformed_X[:, dimensions[1]]
        vectors = np.transpose(clf.components_[dimensions, :])
        vectors_scaled = vectors * [xs.max(), ys.max()]
        for i in range(vectors.shape[0]):
            ax.annotate("", xy=(vectors_scaled[i, dimensions[0]], vectors_scaled[i, dimensions[1]]),
                        xycoords='data', xytext=(0, 0), textcoords='data',
                        arrowprops={'arrowstyle': '-|>', 'ec': 'r'})

            ax.text(vectors_scaled[i, 0] * 1.05, vectors_scaled[i, 1] * 1.05,
                    feature_labels[i] if feature_labels else "Variable" + str(i),
                    color='b', fontsize=text_fontsize)

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(f'Principal Component {dimensions[0]+1}', fontsize=text_fontsize)
    ax.set_ylabel(f'Principal Component {dimensions[1]+1}', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    # Display legend
    ax.legend(
        loc='best',
        shadow=False,
        scatterpoints=1,
        fontsize=text_fontsize
    )
    plt.tight_layout()
    return ax