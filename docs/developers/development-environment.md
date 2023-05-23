# Setting Up VSCode for Python Development

This guide will walk you through the process of setting up Visual Studio Code (VSCode) for Python development with a focus on machine learning.

## Prerequisites

- VSCode installed
- Python installed (the version depends on your project requirements)

## 1. Installing Essential VSCode Extensions

VSCode provides a rich ecosystem of extensions that can make Python development smoother and more efficient. Here are some key extensions to install:

1. **Python** (`ms-python.python`) - Offers Python language support including IntelliSense, linting, debugging, code formatting, etc.
2. **Jupyter** (`ms-toolsai.jupyter`) - Provides Jupyter notebook support, interactive programming and computing.
3. **Python Test Explorer** (`LittleFoxTeam.vscode-python-test-adapter`) - Supports unit testing in Python.


To install an extension, follow these steps:

- Press `Ctrl+Shift+X` to open the Extensions view.
- Type the name of the extension in the search bar.
- Click 'Install'.

## 2. Setting Up the Python Environment

VSCode needs to be configured to use the correct Python interpreter for your project. To do this:

- Click on the Python version displayed in the VSCode status bar; this will ONLY appear when a Python file is open.
- Select the Python interpreter you want to use.

You can also configure the interpreter used by VSCode by modifying the `.vscode/settings.json` file in your workspace:

```json
{
    "python.defaultInterpreterPath": "/path/to/your/python"
}
```

*NOTE*: When setting the interpreter via the UI, VSCode saves your selection in a workspace-specific sqlite settings file under `~/.config/Code/User/workspaceStorage/`. That is intended to be a way to share settings but is ignored when you select the interpreter via the UI.

## 3. Configuring VSCode for Debugging and Tracing

To set up debugging in Python with VSCode, see the [Troubleshooting Guide](./troubleshooting.md#visual-studio-codes-debugger).

## 4. Enabling Pair Programming in VSCode

To enable pair programming, install the "Live Share" extension. This allows you to share your workspace with others for collaborative work.

## 5. Configuring Jupyter Notebook Support in VSCode

With the Jupyter extension installed, you can create a new Jupyter notebook by clicking on the new file button in the Explorer view and giving the file a .ipynb extension.

With these steps, you will have a robust and efficient Python development environment set up in VSCode.
