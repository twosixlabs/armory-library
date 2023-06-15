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

PROJECT_DIR := $(shell git rev-parse --show-toplevel)/examples
SRC_DIR     := $(PROJECT_DIR)/src
BUILD_DIR   := $(PROJECT_DIR)/dist

default: $(.DEFAULT_GOAL)
all: install build


######################
# Functions
######################
define CreateVirtualEnv
	echo "🐍 Setting up virtual environment..."
	ARMORY_CI_TEST=${ARMORY_CI_TEST:-0}

	if [[ "${ARMORY_CI_TEST}" -ne 1 ]]; then
		if [ ! -d venv ]; then
			python -m venv --copies venv
		fi
		source venv/bin/activate
	fi
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


.PHONY: build
build: ## Build the application
	pip install --upgrade wheel
	hatch build --clean --target wheel


.PHONY: clean
clean: ## Remove build, test, and other Python artifacts
	rm -rf .cache

