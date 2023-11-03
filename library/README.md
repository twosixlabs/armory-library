# Overview

Charmory is a scaffolding name as we rework code coming from the `armory.` namespace.
It is slated to be renamed to `armory` once we adapt all legacy code that needs
to be adapted. We expect the `charmory.` namespace to be disappear by the end of 2023.

Presently, working use of armory-library, as shown in the `examples/` directory
imports symbols from both `armory` and `charmory` namespaces. Soon a global substitution
in user code from `charmory` to simply `armory` will be needed. We'll announce
in the release notes when this is needed.



# Installation & Configuration

```bash
pip install armory-library
```

Will make the `armory` and `charmory` namespaces available to your Python environment.

