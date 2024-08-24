.. raw:: html

  <style>
    /* h3 headings on this page are the questions; make them rubric-like */
    h3 {
      font-size: 1rem;
      font-weight: bold;
      padding-bottom: 0.2rem;
      margin: 2rem 0 1.15rem 0;
      border-bottom: 1px solid var(--pst-color-border);
    }

    /* Increase top margin for first question in each section */
    h2 + section > h3 {
      margin-top: 2.5rem;
    }

    /* Make the headerlinks a bit more visible */
    h3 > a.headerlink {
      font-size: 0.9rem;
    }

    /* Remove the backlink decoration on the titles */
    h2 > a.toc-backref,
    h3 > a.toc-backref {
      text-decoration: none;
    }
  </style>

.. _faq:

==========================
Frequently Asked Questions
==========================

.. currentmodule:: scikiplot

Here we try to give some answers to questions that regularly pop up on the mailing list.

.. contents:: Table of Contents
  :local:
  :depth: 2


About the project
-----------------

What is the project name (a lot of people get it wrong)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scikit-plots, but not scikit or SciKit nor sci-kit plots.
Also not scikits.plots or scikits-plots, which were previously used.

How do you pronounce the project name?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sy-kit plots. sci stands for plots!

Why scikit?
^^^^^^^^^^^
There are multiple scikits, which are scientific toolboxes built around SciPy.
Apart from scikit-plots _.

Do you support PyPy?
^^^^^^^^^^^^^^^^^^^^

Due to limited maintainer resources and small number of users, using
scikit-plots with `PyPy <https://pypy.org/>`_ (an alternative Python
implementation with a built-in just-in-time compiler) is not officially
supported.