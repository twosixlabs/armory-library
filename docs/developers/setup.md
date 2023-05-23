# Software Setup Guide

This guide will walk you through setting up your development environment on both Linux and Windows operating systems.

## Git Configuration & Setup

### Configuring Git

Make sure to follow GitLab's guide for [Using SSH Keys to communicate with GitLab](https://docs.gitlab.com/ee/user/ssh.html).

Instructions for setting up Git on Ubuntu Linux are as follows:
  * Install Git with the following command:

      ```bash
      sudo apt-get install git
      ```

  * Configure Git to handle line endings with the following command:

      ```bash
      git config --global core.autocrlf input
      ```

All of these configurations set Git to normalize line endings to LF on commit, and on Windows, to convert them back to CRLF when files are checked out.

### Initial Setup
1. **Clone the repository**
  ```bash
  git clone git@gitlab.jatic.net:jatic/twosix/armory.git
  cd armory
  ```

2. **Install the package**
  ```bash
  make install
  ```

---

## Prerequisites

- Python installed (the version depends on your project requirements)
- VSCode installed
- Git

## Virtual Environment Setup and Project Installation

We use Python's built-in `venv` module to create virtual environments. This keeps our project's dependencies isolated from other Python projects.

In a terminal, navigate to your project directory and run the following commands:

```bash
# Create a virtual environment
python -m venv --copies venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip, and install the build and wheel packages
python -m pip install --upgrade pip build wheel

# Install the project with all dependencies, without compiling
pip install --no-compile --editable '.[all]'
```

## Linting the Code

Before committing any code, we run a pre-commit script that lints the code to ensure it meets our coding standards:

```bash
./tools/pre-commit.sh
```

## Building the Application

We use hatch to build our Python application into a wheel file:

```bash
make build
```

## Generating Documentation

We use mkdocs to build our project documentation:

```bash
make docs
```

---

# Helpful VSCode Tips

  - You can select the Python interpreter used by VSCode by clicking on the Python version on the bottom-left of the status bar.
  - Use the Ctrl+Shift+P shortcut to open the Command Palette, where you can access all VSCode commands.
  - Use the Ctrl+ shortcut to increase the size of the text in the editor.
  - You can split your editor into multiple panes to work with multiple files at the same time. Use the Ctrl+\ shortcut to split the editor.
  - If you want to run a Python file in the terminal, you can right-click anywhere in the file and select 'Run Python File in Terminal'.


Please don't hesitate to ask if you have any questions about our software setup process. We're here to help!
