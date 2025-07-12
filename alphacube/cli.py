"""
Command-Line Interface for the AlphaCube Solver.

This module provides the entry point for the command-line utility, allowing users
to solve Rubik's Cubes directly from the terminal. It parses arguments for
model selection, scramble input, search parameters, and verbosity, then
invokes the core solver and prints the result.
"""
import argparse

import rich


def main():
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
    from . import device, load, set_verbose, solve
    from ._validator import Input

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

    # Pydantic validation
    args_valid = Input(
        format=args.format,
        scramble=args.scramble,
        beam_width=args.beam_width,
        extra_depths=args.extra_depths,
        ergonomic_bias=None,
    )

    load(args.model_id)
    solutions = solve(
        scramble=args_valid.scramble,
        format=args_valid.format,
        beam_width=args_valid.beam_width,
        extra_depths=args_valid.extra_depths,
    )
    rich.print(solutions)
