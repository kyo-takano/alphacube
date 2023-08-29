"""
Solve Manager

This module provides a Solver class for finding Rubik's Cube solutions using a beam search algorithm.

Class:

- ``Solver``: A class for managing Rubik's Cube configuration, solving model, and search function.

Example::

    from alphacube.solver import Solver
    solver = Solver() # Assigned to `alphacube._solver` at the package level. 
    solver.load()
    solution = solver(format='moves', scramble="R U R' U'", beam_width=1024)

    """

import torch
from . import logger, logargs
from .env import Cube3
from .search import beam_search
from .model import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class Solver:
    """
    A solver class for managing environment, model, and search configurations.
    Methods:
    - ``load``: Load the solver model and optimize it for CPU or GPU.
    - ``__call__``: Set up the cube state and pass it for solution using beam search.
    """
    def load(
        self, 
        prefer_gpu=True, 
        quantize_on_cpu=True, 
        jit_mode=False, 
        *args, **kwargs
    ):
        """
        Load the Rubik's Cube solver model and optimize it for CPU or GPU.

        Args:
            prefer_gpu (bool): Whether to prefer GPU if available.
            quantize_on_cpu (bool): Whether to quantize the model for CPU optimization.
            jit_mode (bool): Whether to enable JIT mode for potentially faster execution.
            *args: Additional arguments for model loading.
            **kwargs: Additional keyword arguments for model loading and optimization.

        Returns:
            None
        """
        # Load the model (download if not yet)
        self.model = load_model(*args, **kwargs)
        if prefer_gpu and DEVICE != "cpu":
            logger.info(f"[grey50]Running on {DEVICE.upper()}", **logargs)
            self.model.to(DEVICE)
        else:
            logger.info(f"[grey50]Running on CPU (no GPU found)", **logargs)
            if quantize_on_cpu:
                logger.info(f"[grey50]Quantizing the model -- roughly 3x faster [italic]on CPU", **logargs)
                self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

        if jit_mode:
            logger.info(f"[grey50]JIT-mode enabled -- [italic]potentially[/italic] faster than eager execution", **logargs)
            self.model = torch.jit.script(self.model)

        logger.info(f"[cyan]Initialized AlphaCube solver.", **logargs)


    def __call__(self, scramble, format="moves", allow_wide=True, **kwargs):
        """
        Set up the cube state from `format` and `scramble` and pass it to ``search.beam_search`` together with ``**kwargs``.

        Args:
            format (str): Input format of the scramble: either "moves" or "stickers".
            scramble (list): A sequence of moves/stickers representing the initial state of the Rubik's Cube.
            allow_wide (bool): Whether wide moves are allowed.
            **kwargs: Keyword arguments to be passed to ``beam_search``.

        Returns:
            solutions (dict | None): Dictionary containing the solution response from ``search.beam_search``.
        """

        env = Cube3(allow_wide=allow_wide)

        if format == "moves":
            env.apply_scramble(scramble)
        elif format == "stickers":
            env.state[:] = scramble
        else:
            raise ValueError("Unexpected value for `format`")

        # Reset sticker colors if not right
        env.reset_axes()

        if env.is_solved():
            return ValueError("Looks like it is already solved.")

        return beam_search(env, self.model, **kwargs)