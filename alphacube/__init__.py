"""
`alphacube` package provides a few functions for different purposes:

- `load(*args, **kwargs)`: Load a trained DNN.
- `solve(*args, **kwargs)`: Solve a Rubik's Cube using the loaded solver.
- `set_verbose(loglevel=logging.INFO)`: Set the verbosity level of the logger.
- `cli()`: Command-line utility for solving a Rubik's Cube using AlphaCube.

See [Getting Started](../getting-started) for the basic usage.
"""

from .utils import logger, logargs, set_verbose, device, dtype

from ._validator import Input
from .core import Solver


solver = _solver = Solver()

__all__ = ["load", "solve", "cli", "solver", "logger", "logargs", "set_verbose", "device", "dtype"]


def load(model_id: str = "small" if device.type == "cpu" else "large", *args, **kwargs):
    """
    Load the Rubik's Cube solver model.

    Args:
        model_id: Identifier for the model variant to load ("small", "base", or "large"). Default to `small` on CPU; `large` on GPU/MPS.
        *args: Additional positional arguments to configure model loading.
        **kwargs: Additional keyword arguments to configure model loading.

    Returns:
        None
    """
    solver.load(model_id, *args, **kwargs)


def solve(*args, **kwargs):
    """
    Solve a Rubik's Cube puzzle using the loaded solver model.

    Args:
        *args, **kwargs: Arguments to configure puzzle solving; passed down to `alphacube.core.Solver.__call__()` and `alphacube.search.beam_search`.

    Returns:
        dict | None: A dictionary containing solutions and performance metrics. None if failed.
    """
    if not hasattr(solver, "model"):
        raise ValueError("Model not loaded. Call `load`  first.")

    return solver(*args, **kwargs)


def list_models():
    """
    List the available model IDs.

    Returns:
        list: A list of available model IDs.
    """
    return ["small", "base", "large"]


# CLI Option
def cli():
    """
    Command-line utility for solving a Rubik's Cube using AlphaCube.

    ```bash title="Syntax"
    alphacube [--model_id MODEL_ID] [--format FORMAT] [--scramble SCRAMBLE]
            [--beam_width BEAM_WIDTH] [--extra_depths EXTRA_DEPTHS]
    ```

    Arguments:

    * `--model_id`/`-m` (str): Choose a specific model for solving (default: 'small' on CPU, otherwise `large`; another choice is `base``).
    * `--format`/`-f` (str): Specify the input format ('moves' or 'stickers').
    * `--scramble`/`-s` (str): Define the initial cube state using either a sequence of moves or a stringify JSON dictionary.
    * `--beam_width`/`-bw` (int): Set the beam width for search (default: 1024).
    * `--extra_depths`/`-ex` (int): Specify additional depths for exploration (default: 0).
    * `--verbose`/`-v`: Enable verbose output for debugging and tracking progress.

    Returns:
        None

    Example Usages:

    ```bash title="1. Solve a cube using default settings"
    alphacube --scramble "R U R' U'"
    ```

    ```bash title="2. Solve a cube with custom settings"
    alphacube --model_id large --beam_width 128 --extra_depths 2 \\
            --scramble "R U2 F' R2 B R' U' L B2 D' U2 R F L"
    ```

    ```bash title="3. Solve a cube using a sticker representation"
    alphacube --format stickers \\
            --scramble '{ \\
                "U": [0, 0, 5, 5, 0, 5, 5, 4, 0], \\
                "D": [1, 3, 3, 4, 1, 1, 4, 1, 3], \\
                "L": [4, 5, 1, 0, 2, 2, 0, 1, 4], \\
                "R": [5, 2, 0, 2, 3, 3, 2, 0, 2], \\
                "F": [4, 3, 2, 4, 5, 1, 1, 4, 3], \\
                "B": [1, 5, 5, 0, 4, 3, 3, 2, 2] \\
            }' \\
            --beam_width 64
    ```
    """
    import argparse

    from rich import print as rprint

    parser = argparse.ArgumentParser(
        description="AlphaCube -- State-of-the-Art Rubik's Cube Solver"
    )
    parser.add_argument(
        "--model_id",
        "-m",
        type=str,
        default="small" if device.type == "cpu" else "large",
        help="ID of the model to solve a given scramble with (`small`, `base`, or `large`)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="moves",
        help="Format of the input scramble (`moves` or `stickers`)",
    )
    parser.add_argument(
        "--scramble",
        "-s",
        type=str,
        help="Sequence of scramble in HTM (including wide moves) or dictionary of sticker indices by face.",
    )
    parser.add_argument(
        "--beam_width",
        "-bw",
        type=int,
        default=1024,
        help="Beam width, the parameter strongly correlated with solution optimality and computation time",
    )
    parser.add_argument(
        "--extra_depths",
        "-ex",
        type=int,
        default=0,
        help="Number of additional depths to explore during the search",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="loglevel",
        action="store_const",
        const=20,
        help="Enable verbose output for tracking progress",
    )
    args = parser.parse_args()

    if args.loglevel:
        set_verbose(args.loglevel)

    # Manual validation
    model_id = args.model_id
    assert model_id in ["small", "base", "large"], "Model ID must be either 'base' or 'large'"

    logger.info(args)

    # Pydantic validation
    args = Input(
        format=args.format,
        scramble=args.scramble,
        beam_width=args.beam_width,
        extra_depths=args.extra_depths,
        ergonomic_bias=None,
    )

    # Load only once validated
    solver.load(model_id)
    # Solve
    solutions = solver(
        format=args.format,
        scramble=args.scramble,
        beam_width=args.beam_width,
        extra_depths=args.extra_depths,
    )
    rprint(solutions)
