## Makefile
#### This Makefile contains various targets for project management tasks such as running the project,
### cleaning up build files, running tests, building Docker images, and more.
## Phony targets are used to avoid conflicts with files of the same name.


## Declare phony targets to indicate these are not files but commands to be executed.
.PHONY: clean examples test publish all
## To run any of these targets, use the `make` command followed by the target name.
## For example:
##   make clean            # Executes the clean target to remove build artifacts
##   make examples         # Executes the examples target to run code save generated plots.
##   make test             # Executes the test target to run unit tests
##   make publish          # Executes the publish target to run unit tests and publish pypi
##   make all              # Executes the all target to clean and build the project


## clean target: Removes build artifacts and cleans up the project directory.
## Useful for ensuring a fresh build environment.
clean:
	# mkdir -p current_dir
	rm -rf `find -L -type d -name .ipynb_checkpoints`
	rm -rf `find -L -type d -name __pycache__` .pytest_cache
	rm -rf docs/build
	rm -rf build dist scikit_plots.egg-info
	echo "clean completed."


## example_script target: Runs py script on the examples/ directory.
## Run this target to save generated script plot image.
examples:
	cd examples && python plot_calibration_curve.py
	cd examples && python plot_classifier_eval.py
	cd examples && python plot_confusion_matrix.py
	cd examples && python plot_cumulative_gain.py
	cd examples && python plot_elbow_curve.py
	cd examples && python plot_feature_importances.py
	cd examples && python plot_ks_statistic.py
	cd examples && python plot_learning_curve.py
	cd examples && python plot_lift.py
	cd examples && python plot_pca_2d_projection.py
	cd examples && python plot_pca_component_variance.py
	cd examples && python plot_precision_recall.py
	cd examples && python plot_roc.py
	cd examples && python plot_silhouette.py
	echo "All py Script executed."


## test target: Runs pytest on the tests/ directory.
## Run this target to execute unit tests.
test: clean
	pytest tests/
	echo "pytest completed."


## publish target: Builds the pypi Packages, and publishes the library.
## This target depends on clean and test.
publish: test clean
	python -m build
	twine upload dist/*
	echo "pypi publish completed."


# all target: A convenience target that cleans the build directory and then builds the app.
# Ensures that the project is rebuilt from a clean state.
all: clean
	# Assuming my_app is built using some other rule or command
	make my_app
	echo "all completed."