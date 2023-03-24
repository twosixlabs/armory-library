# Getting Started

## Installation
```bash
git clone https://gitlab.jatic.net/jatic/twosix/armory.git
cd armory
pip install --editable ".[all]"
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
