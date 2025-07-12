"""
`alphacube` package provides a flexible API for solving Rubik's Cubes.

High-Level API (recommended for most users):
- `load(*args, **kwargs)`: Convenience function to load a model into a default global solver.
- `solve(*args, **kwargs)`: Convenience function to solve a cube using the default global solver.

Core Class (for advanced usage):
- `Solver`: The main class for creating solver instances, loading models, and solving cubes.

Utilities:
- `set_verbose(loglevel)`: Set the verbosity level of the logger.
- `list_models()`: List available pre-trained models.
- `device`: The auto-detected `torch.device` (e.g., 'cuda', 'cpu').
- `dtype`: The auto-detected `torch.dtype` (e.g., 'torch.float16').
- `cli()`: Command-line utility for solving a Rubik's Cube.

See [Getting Started](https://alphacube.dev/docs/getting-started) for the basic usage.
"""

from .core import Solver
from .utils import logger, set_verbose, list_models, device, dtype
from .cli import main as cli


solver = Solver()

# shortcuts
load = solver.load
solve = solver.__call__

# `_solver` left for backward compatibility
_solver = solver

__all__ = [
    "Solver",
    "solver",
    "_solver",
    "load",
    "solve",
    "cli",
    "logger",
    "set_verbose",
    "list_models",
    "device",
    "dtype",
]
