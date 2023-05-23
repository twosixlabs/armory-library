# Troubleshooting Guide

This guide provides tips and instructions on how to troubleshoot issues in our codebase using Visual Studio Code's debugger and Python's pdb module.

## Visual Studio Code's Debugger

### Configuration

First, configure the launch settings in the `.vscode/launch.json` file in your workspace. Here's a basic configuration:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

### Usage
VSCode's built-in debugger is a powerful tool for finding and fixing issues in your code. Here are some tips on how to use it effectively:

1. **Setting Breakpoints**: Click in the left margin of your code file (next to the line numbers) to set a breakpoint on a specific line. The program will pause its execution at this point, allowing you to inspect its state.

2. **Stepping Through Code**: Once your code execution is paused, you can step through your code using the step over (`F10`), step into (`F11`), and step out (`Shift+F11`) commands in the debugger toolbar.

3. **Inspecting Variables**: While paused at a breakpoint, you can view the values of variables in the "Variables" section of the Run view. You can also hover over variables in your code to see their values.

4. **Watching Expressions**: In the "Watch" section of the Run view, you can add expressions that will be evaluated whenever the code execution is paused. This is useful for keeping an eye on the values of specific variables or expressions.


## Python's pdb

Python's pdb is a built-in module that provides a simple but effective interactive debugger for Python programs. Here's how you can use it in your code:

1. **Starting a pdb Session**: Add the line `breakpoint()` at the point in your code where you want to start the debugger. This will pause the execution of your code at that line and start an interactive pdb session.

2. **Inspecting Variables**: In the pdb session, you can type the name of any variable to see its current value.

3. **Stepping Through Code**: Use the `next` command to execute the current line and move to the next one, or the `step` command to move into the function called on the current line.


## VSCode Tips

- **Go to Definition**: Right-click on a method or variable and select "Go to Definition" to quickly navigate to where it's defined.

- **Find All References**: Right-click on a method or variable and select "Find All References" to see all the places where it's used in your codebase.

- **Quick Open**: Use `Ctrl+P` to quickly open any file in your workspace by typing part of its name.

- **Multi-Cursor Selection**: Hold `Alt` and click in different places in your code to create multiple cursors, allowing you to edit multiple lines at the same time.


Don't hesitate to ask for help if you're stuck on a difficult bug!
