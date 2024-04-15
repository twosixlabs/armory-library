#!/bin/env bash

set -e

if [ -z "$VIRTUAL_ENV" ]; then
    read -p "You are not currently using a virtual environment. Do you want to proceed? " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

set +e
PYTHON=$(which ${PYTHON:-python})
set -e

if [ -z "$PYTHON" ]; then
    echo "Python interpreter not found. Exiting."
    exit 1
fi

echo "Installing all armory-suite components using $PYTHON"
set -x
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --editable .
$PYTHON -m pip install --editable library
$PYTHON -m pip install --editable matrix
$PYTHON -m pip install --editable examples[all]
set +x

read -p "Do you want to install the pre-commit hook? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    set -x
    $PYTHON -m pre_commit install
fi
