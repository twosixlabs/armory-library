---
marp: true
---

# Marp — Markdown Presentation Ecosystem

* Create presentation slides
    1. in markdown format
    2. with almost no effort
    3. and no need to learn a new language
    4. easy support for
        - Linux / Windows / Macos
        - VSCode extension (Ctrl-Shift-V like always)

---

# Marp Syntax - you already know it

* Slides start with a `# header` and end with a `---` horizontal rule.
* Lists and code blocks are standard markdown.
* A preamble is used by Marp to set options. The minimal preamble is
    ```markdown
    ---
    marp: true
    ---
    ```
* Easiest install via [VS Code extension](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
* Run `marp slides.md` to generate `slides.html`
