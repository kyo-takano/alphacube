"""
This module provides the core class `Solver`, which loads a mode and finds Rubik's Cube solutions using a beam search algorithm.

Class:

- `Solver`: A class for managing Rubik's Cube configuration, solving model, and search function.
"""

from .env import Cube3
from .model import load_model
from .search import beam_search
from .utils import logger, device, cache_dir
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
        quantize_on_cpu: bool = True,
        cache_dir: str = cache_dir,
        **kwargs_muted,
    ):
        """
        Load the Rubik's Cube solver model and optimize it for CPU or GPU.

        Args:
            model_id (str): Identifier for the model variant to load ("small", "base", or "large").
            quantize_on_cpu (bool): Whether to quantize the model for CPU optimization.
            cache_dir (str): Directory to cache the model files.

        Returns:
            None
        """
        if "jit_mode" in kwargs_muted:
            logger.warning(
                "The argument `jit_mode` has been muted due to the compilation overhead. "
                "If you still wish to compile the model, we recommend executing something like `torch.compile(alphacube.solver.model)`"
            )
        if "prefer_gpu" in kwargs_muted:
            logger.warning(
                "The argument `prefer_gpu` has been deprecated due to its unnecessary redundancy. "
                "If you are using accelerator and still want to compute on GPU, please move the model by calling `alphacube.solver.model.to('cpu')`"
            )

        # Load the model (download if not yet)
        self.model_id = model_id
        self.model = load_model(
            self.model_id, quantize=quantize_on_cpu and device.type == "cpu", cache_dir=cache_dir
        )

        if device.type == "cpu":
            logger.info("[grey50]Running on CPU (no accelerators found)")
        else:
            logger.info(f"[grey50]Running on {device.type.upper()}")
            self.model.to(device)

        logger.info("[cyan]Initialized AlphaCube solver.")

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
