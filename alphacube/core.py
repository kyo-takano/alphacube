"""
This module provides the core class `Solver`, which loads a mode and finds Rubik's Cube solutions using a beam search algorithm.

Class:

- `Solver`: A class for managing Rubik's Cube configuration, solving model, and search function.
"""

import torch

from .env import Cube3
from .model import load_model
from .search import beam_search
from .utils import logger, logger_args, device, cache_dir
from ._evaluator import benchmark, evaluate_temporal_performance


class Solver:
    """
    A solver class for managing environment, model, and search configurations.

    This class orchestrates the cube environment, the neural network model, and the
    beam search algorithm to find solutions.

    Methods:
    - `load`: Load the solver model and optimize it for CPU or GPU.
    - `__call__` (or `solve`): Set up the cube state and find solutions using beam search.
    - `benchmark`: Evaluate the solver's search efficiency.
    - `evaluate_temporal_performance`: (Deprecated) Evaluate solution length vs. time.
    """

    def load(
        self,
        model_id: str = dict(cpu="small").get(device.type, "large"),
        prefer_gpu: bool = True,
        quantize_on_cpu: bool = True,
        jit_mode: bool = False,
        cache_dir: str = cache_dir,
    ):
        """
        Load the Rubik's Cube solver model and optimize it for CPU or GPU.

        Args:
            model_id (str): Identifier for the model variant to load ("small", "base", or "large").
            prefer_gpu (bool): Whether to prefer GPU if available.
            quantize_on_cpu (bool): Whether to quantize the model for CPU optimization.
            jit_mode (bool): Whether to enable JIT mode for potentially faster execution.
            cache_dir (str): Directory to cache the model files.

        Returns:
            None
        """
        # Load the model (download if not yet)
        self.model_id = model_id
        self.model = load_model(self.model_id, cache_dir=cache_dir)
        if prefer_gpu and device.type != "cpu":
            logger.info(f"[grey50]Running on {device.type.upper()}", **logger_args)
            self.model.to(device)
        else:
            logger.info("[grey50]Running on CPU (no GPU found)", **logger_args)
            if quantize_on_cpu:
                logger.info(
                    "[grey50]Quantizing model for CPU execution. "
                    "This should provide a significant speedup (~3x).",
                    **logger_args,
                )
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )

        if jit_mode:
            logger.info(
                "[grey50]JIT compilation enabled. This may improve performance over standard eager execution.",
                **logger_args,
            )
            self.model = torch.jit.script(self.model)

        logger.info("[cyan]Initialized AlphaCube solver.", **logger_args)

    def __call__(
        self,
        scramble,
        format="moves",
        allow_wide=True,
        # **kwargs,
        beam_width: int = 1024,
        extra_depths: int = 0,
        ergonomic_bias: dict | None = None,
    ):
        """
        Set up the cube state and find solutions using beam search.

        Args:
            scramble (list or str): A sequence of moves or a sticker representation
                of the initial cube state.
            format (str): Input format of the scramble: "moves" or "stickers".
            allow_wide (bool): Whether to allow wide moves in the search space.
                Note: This is often controlled automatically by `ergonomic_bias`.
            beam_width (int): The beam width for the search algorithm.
            extra_depths (int): Number of extra depths to search after finding a solution.
            ergonomic_bias (dict, optional): A dictionary to bias the search towards
                certain moves based on ergonomic preference.

        Returns:
            dict | None: A dictionary containing the solution(s) and search metadata,
                        or None if no solution is found.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model not loaded. Call `load` first.")

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

        return beam_search(
            env,
            self.model,
            # **kwargs
            beam_width=beam_width,
            extra_depths=extra_depths,
            ergonomic_bias=ergonomic_bias,
        )

    # fmt:off
    benchmark = evaluate_search_efficiency = benchmark    # A shortcut method `alphacube.solver.benchmark(...)` functionally equivalent to `alphacube._evaluator.benchmark(alphacube.solver, ...)`. `evaluate_search_efficiency` redirects to this method for backward compatibility.
    # fmt:on

    evaluate_temporal_performance = evaluate_temporal_performance  # Deprecated. A shortcut `alphacube._evaluator.evaluate_temporal_performance`
