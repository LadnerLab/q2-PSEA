.PHONY: all lint test test-cov install dev clean distclean

PYTHON ?= python

all: ;

lint:
	flake8 \
	--exclude ./build/,./versioneer.py,./setup.py

test: all
	py.test

test-cov: all
	py.test --cov=q2_epitope

install: all
	$(PYTHON) -m pip install -v .

dev: all
	pip install -e .

clean: distclean

distclean: ;
