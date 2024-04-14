PYTHON_VERSION := 3.11
PYTHON_BIN := python$(PYTHON_VERSION)
PACKAGE_INDEX := https://pypi.org/simple
VENV_DEV := .venv-dev
VENV_CI := .venv-ci

$(VENV_DEV):
	$(PYTHON_BIN) -m venv $(VENV_DEV)
	. $(VENV_DEV)/bin/activate; \
	    pip3 config --site set global.index-url "$(PACKAGE_INDEX)"; \
		pip3 install --upgrade pip; \
		pip3 install -e .


$(VENV_CI):
	$(PYTHON_BIN) -m venv $(VENV_CI)
	. $(VENV_CI)/bin/activate; \
		pip3 config --site set global.index-url "$(PACKAGE_INDEX)"; \
        pip3 install --upgrade pip; \
		pip3 install \
			build \
			tox \
			twine \

.PHONY: setup-dev
setup-dev: $(VENV_DEV)

.PHONY: test
test: $(VENV_CI)
	. $(VENV_CI)/bin/activate; \
    		tox

dist: $(VENV_CI) pyproject.toml $(shell find src/timeseries_etl_utils -type f)
    # Deletes the dist directory and all contents
	rm -rf $@
	. $(VENV_CI)/bin/activate; \
		python3 -m build

.PHONY: build-python
build-python: dist


.PHONY: publish-python-test
publish-python-test: build-python
	. $(VENV_CI)/bin/activate; \
		twine upload \
		--repository testpypi \
		dist/*

.PHONY: publish-python
publish-python: build-python
	. $(VENV_CI)/bin/activate; \
		twine upload \
		dist/*


.PHONY: clean
clean:
	rm -rf \
		$(VENV_DEV) \
		$(VENV_CI) \
		dist/ \
		$(shell find . -iname '*.egg-info' -type d) \
		$(shell find . -iname '__pycache__' -type d) \
		$(shell find . -iname '.pytest_cache' -type d) \
		.mypy_cache \
		coverage-reports/ \
		xunit-reports/ \
		.tox/ \
		.scannerwork/