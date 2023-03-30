#!/usr/bin/make -f
# -*- makefile -*-

SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c
.DEFAULT_GOAL := help
.LOGGING      := 0

.ONESHELL:             ;	# Recipes execute in same shell
.NOTPARALLEL:          ;	# Wait for this target to finish
.SILENT:               ;	# No need for @
.EXPORT_ALL_VARIABLES: ;	# Export variables to child processes.
.DELETE_ON_ERROR:      ;	# Delete target if recipe fails.

# Modify the block character to be `-\t` instead of `\t`
ifeq ($(origin .RECIPEPREFIX), undefined)
	$(error This version of Make does not support .RECIPEPREFIX.)
endif
.RECIPEPREFIX = -

PROJECT_DIR := $(shell git rev-parse --show-toplevel)
SRC_DIR     := $(PROJECT_DIR)/src
BUILD_DIR   := $(PROJECT_DIR)/dist

default: $(.DEFAULT_GOAL)
all: install lint scan build docs


######################
# Functions
######################
define Install
	echo "🐍 Setting up virtual environment..."
	if [ ! -d "./venv" ]; then
		mkdir -p .cache/
		python3 -m venv --copies "./venv"
	fi
	source "./venv/bin/activate"
	python3 -m pip install --upgrade pip build wheel
	pip3 install --no-compile --editable '.[all]'
endef


define TypeCheck
	python3 -m mypy src          \
		--ignore-missing-imports   \
		--follow-imports=skip      \
		--show-error-codes         \
		--show-column-numbers      \
		--pretty
endef


######################
# Commands
######################
.PHONY: help
help: ## List commands <default>
-	echo -e "USAGE: make \033[36m[COMMAND]\033[0m\n"
-	echo "Available commands:"
-	awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\t\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


.PHONY: install
install:	## Setup a Virtual Environment
-	$(call Install)


.PHONY: lint
lint: ## Run style checks
-	ARMORY_CI_TEST=1 ./tools/pre-commit.sh


.PHONY: scan
scan: ## Run bandit security scan
-	python3 -m bandit -v -f txt -r ./src -c "pyproject.toml" --output bandit_scan.txt || exit 0
-	$(call TypeCheck)


.PHONY: test
test: ## Run application tests
-	armory
# pytest -c pyproject.toml -s ./tests.orig/unit/test_configuration.py
# pytest -c pyproject.toml -m "not docker_required and unit" ./tests.orig/
# pytest -c pyproject.toml -s ./tests.orig/end_to_end/test_no_docker.py


.PHONY: type
type: ## Type check the code
-	$(call TypeCheck)


.PHONY: update
update: ## git pull branch
-	git pull origin `git config --get remote.origin.url`


.PHONY: pip-update
pip-update: ## Update pip packages
-	pip install --upgrade $(pip freeze | awk -F'[=]' '{print $1}')


.PHONY: build
build: ## Build the application
-	pip3 install --upgrade wheel
-	hatch build --clean --target wheel


.PHONY: docs
docs: ## Create documentation
-	mkdocs build --clean


.PHONY: clean
clean: ## Remove build, test, and other Python artifacts
-	rm -rf .cache
