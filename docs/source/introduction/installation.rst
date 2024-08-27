.. _installation:

=======================
Installing scikit-plots
=======================

Scikit-plots depends on `Scikit-learn <http://scikit-learn.org/>`_ and `Matplotlib <http://matplotlib.org/>`_ to do its magic, so make sure you have them installed as well.

There are different ways to install scikit-learn:

* :ref:`Install the latest official release <install_official_release>`. This
  is the best approach for most users. It will provide a stable version
  and pre-built packages are available for most platforms.

* Install the version of scikit-learn provided by your
  :ref:`operating system or Python distribution <install_by_distribution>`.
  This is a quick option for those who have operating systems or Python
  distributions that distribute scikit-learn.
  It might not provide the latest release version.

* :ref:`Building the package from source
  <install_bleeding_edge>`. This is best for users who want the
  latest-and-greatest features and aren't afraid of running
  brand-new code. This is also needed for users who wish to contribute to the
  project.


.. _install_official_release:

Installing the latest release
=============================

.. raw:: html

  <style>
    /* Show caption on large screens */
    @media screen and (min-width: 960px) {
      .install-instructions .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .install-instructions .sd-tab-set.tabs-package-manager::before {
        content: "Package Manager";
      }

      .install-instructions .sd-tab-set.tabs-os::before {
        content: "Operating System";
      }
    }
  </style>

.. div:: install-instructions

  .. tab-set::
    :class: tabs-package-manager

    .. tab-item:: pip
      :class-label: tab-6

      .. tab-set::
        :class: tabs-os

        .. tab-item:: Windows
          :class-label: tab-4
          :sync: package-manager-pip

          Install the 64-bit version of Python 3, for instance from the
          `official website <https://www.python.org/downloads/windows/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install scikit-learn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: powershell

            python -m venv sklearn-env
            sklearn-env\Scripts\activate  # activate
            pip install -U scikit-learn

          In order to check your installation, you can use:

          .. prompt:: powershell

            python -m pip show scikit-learn  # show scikit-learn version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: Linux
          :class-label: tab-4
          :sync: package-manager-pip

          Python 3 is usually installed by default on most Linux distributions. To
          check if you have it installed, try:

          .. prompt:: bash

            python3 --version
            pip3 --version

          If you don't have Python 3 installed, please install `python3` and
          `python3-pip` from your distribution's package manager.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install scikit-learn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash

            python3 -m venv sklearn-env
            source sklearn-env/bin/activate  # activate
            pip3 install -U scikit-learn

          In order to check your installation, you can use:

          .. prompt:: bash

            python3 -m pip show scikit-learn  # show scikit-learn version and location
            python3 -m pip freeze             # show all installed packages in the environment
            python3 -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: MacOS
          :class-label: tab-4
          :sync: package-manager-pip

          Install Python 3 using `homebrew <https://brew.sh/>`_ (`brew install python`)
          or by manually installing the package from the `official website
          <https://www.python.org/downloads/macos/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install scikit-learn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packges.

          .. prompt:: bash

            python -m venv sklearn-env
            source sklearn-env/bin/activate  # activate
            pip install -U scikit-learn

          In order to check your installation, you can use:

          .. prompt:: bash

            python -m pip show scikit-learn  # show scikit-learn version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import sklearn; sklearn.show_versions()"

    .. tab-item:: conda
      :class-label: tab-6

      .. include:: installation_instructions_conda.rst


Using an isolated environment such as pip venv or conda makes it possible to
install a specific version of scikit-learn with pip or conda and its dependencies
independently of any previously installed Python packages. In particular under Linux
it is discouraged to install pip packages alongside the packages managed by the
package manager of the distribution (apt, dnf, pacman...).

Note that you should always remember to activate the environment of your choice
prior to running any Python command whenever you start a new terminal session.

If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).