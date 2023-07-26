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

# # Modify the block character to be `-\t` instead of `\t`
# ifeq ($(origin .RECIPEPREFIX), undefined)
# 	$(error This version of Make does not support .RECIPEPREFIX.)
# endif
# .RECIPEPREFIX = -

PROJECT_DIR := $(shell git rev-parse --show-toplevel)
SRC_DIR     := $(PROJECT_DIR)/src
BUILD_DIR   := $(PROJECT_DIR)/dist

default: $(.DEFAULT_GOAL)
all: install lint scan test build docs


######################
# Functions
######################
define CreateVirtualEnv
	echo "🐍 Setting up virtual environment..."
	ARMORY_CI_TEST="${ARMORY_CI_TEST:-0}"

	if [ "${ARMORY_CI_TEST}" -ne 1 ]; then
		if [ ! -d venv ]; then
			python -m venv --copies venv
		fi
		source venv/bin/activate
	fi
	python -m pip install --upgrade pip build wheel
	pip install --no-compile --editable '.[all]'
endef


define ExecuteTests
	echo "🤞 Executing tests... Good luck! 🌈"

	PYTEST_PARAMS="--exitfirst --suppress-no-test-exit-code"

	echo "🤞 Executing configuration tests..."
	python -m pytest -c pyproject.toml -s ./tests/test_attack_object.py

	echo "🤞 Executing unit tests..."
	python -m pytest -c pyproject.toml -m "unit" ./tests/unit/
endef


define TypeCheck
	python -m mypy src           \
		--ignore-missing-imports   \
		--follow-imports=skip      \
		--show-error-codes         \
		--show-column-numbers      \
		--pretty
endef


define ProvisionHost
	echo "⏳ Host provisioning started..."

	DEBIAN_FRONTEND=noninteractive

	echo "⏳ Updating packages..."
	apt-get update -qqy
	apt-get install -qqy      \
		--no-install-recommends \
		--no-install-suggests   \
			git                   \
			libgl1-mesa-glx       \
			gnupg                 \
			build-essential       \
			rsync                 \
			make                  \
			software-properties-common
endef


define DockerInstall
	git config --global --add safe.directory /app

	echo "⏳ Installing package..."
	python -m venv --copies .venv
	source .venv/bin/activate

	python -m pip install --upgrade pip build wheel
	pip install --no-compile --editable '.[all]'
endef


######################
# Commands
######################
.PHONY: help
help: ## List commands <default>
	echo -e "USAGE: make \033[36m[COMMAND]\033[0m\n"
	echo "Available commands:"
	awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\t\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


.PHONY: install
install: ## Setup a Virtual Python Environment
	$(call CreateVirtualEnv)


.PHONY: lint
lint: ## Run style checks
	echo "🦄 Linting code..."
	python -m pre_commit run --all-files


.PHONY: scan
scan: ## Run bandit security scan
	echo "🤖 Running security scans... beep boop"
	python -m bandit -v -f txt -r ./src -c "pyproject.toml" --output bandit_scan.txt || exit 0
	$(call TypeCheck)


.PHONY: test
test: ## Run application tests
	$(call ExecuteTests)


.PHONY: build
build: ## Build the application
	pip install --upgrade wheel
	hatch build --clean --target wheel


.PHONY: docs
docs: ## Create documentation
	mkdocs build --verbose --config-file ./tools/mkdocs.yml


.PHONY: clean
clean: ## Remove build, test, and other Python artifacts
	rm -rf .cache


######################
# Developer Tools
######################
.PHONY: update
update: ## git pull branch
	git pull origin `git config --get remote.origin.url`


.PHONY: pip-update
pip-update: ## Update pip packages
	pip install --upgrade $(pip freeze | awk -F'[=]' '{print $1}')


.PHONY: type
type: ## Type check the code
	$(call TypeCheck)


.PHONY: docker
docker: ## Build a container with armory installed
	docker build --tag armory-jatic --file ./tools/Dockerfile.debug .


.PHONY: docker-local
docker-local: ## Start a container with armory installed
	docker pull python:3.8-slim-bullseye
	docker build --tag armory-jatic:latest --file ./tools/Dockerfile.debug .
	docker run -it --rm armory-jatic:latest
#	docker run -it --rm -v `pwd`:/app armory-jatic


.PHONY: docker-install
docker-install: ## Package installation for Ubuntu
	$(call DockerInstall)
