# IMAGINE

A friendly greeting module.

## Quick Start

```python
from salut import salut, greet

salut()            # "Salut !"
salut("Alice")     # "Salut, Alice !"
greet(lang="en")   # "Hello !"
```

Or from the command line:

```bash
python -m salut          # Salut !
python -m salut Alice    # Salut, Alice !
```

## Running Tests

```bash
python -m pytest tests/ -v
```