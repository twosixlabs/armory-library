# Getting Started

## Installation
```bash
pip install armory-library
```

## Usage
### Example:
```python
from charmory.blocks import cifar10, mnist  # noqa: F401
from charmory.engine import Engine

baseline = cifar10.baseline
result = Engine(baseline).run()

print(result)
```

### Example testing entrypoint:
```bash
$ armory
```
