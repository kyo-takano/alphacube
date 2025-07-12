"""
Module for evaluating the performance of a solver.

This module provides functions to evaluate the performance of a solver on a dataset.
It includes functions to evaluate the search efficiency and temporal performance of a solver.

NOTE: The functionality in this module requires optional dependencies. To use them,
please install AlphaCube with the 'eval' extra: `pip install 'alphacube[eval]'`
"""

import json
import os
import warnings

# Core dependencies are imported directly
import numpy as np
import requests
from tqdm import tqdm

from .utils import logger, logger_args, device, cache_dir


# --- Handle Optional Dependencies ---
# Define a clear error message for the user.
_EVAL_DEPS_ERROR_MSG = (
    "Evaluation dependencies are not installed. To use this function, "
    "please install them by running: pip install 'alphacube[eval]'"
)

try:
    # These imports will fail if the [eval] dependencies are not installed.
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    _EVAL_DEPS_INSTALLED = True
except ImportError:
    _EVAL_DEPS_INSTALLED = False
    # We don't need to do anything else. The functions below will handle it.


def get_dataset(filename="deepcubea-dataset--cube3.json", cache_dir=cache_dir):
    """
    Get a dataset from a file or download it if it doesn't exist.

    Args:
        filename (str): The filename of the dataset.

    Returns:
        dict: The dataset.
    """
    filepath = os.path.join(cache_dir, filename)
    if not os.path.exists(filepath):
        os.makedirs(cache_dir, exist_ok=True)
        with requests.get(
            os.path.join("https://storage.googleapis.com/alphacube/", filename), stream=True
        ) as r:
            with open(filepath, "wb") as output:
                for chunk in r.iter_content(chunk_size=8192):
                    output.write(chunk)

    logger.info(f"[grey50]Saved to {filepath}", **logger_args)
    try:
        with open(filepath) as f:
            dataset = json.load(f)
    except Exception as e:
        os.remove(filepath)
        raise ValueError(
            f"Failed to load the dataset from '{filepath}'. The file might be corrupt. "
            f"The cached dataset has been deleted to allow for a fresh download. Original error: {e}"
        )

    return dataset


def evaluate_search_efficiency(
    solver,
    num_samples=1000,
    beam_width=2**10 if device.type == "cpu" else 2**13,
    verbose=False,
):
    """
    Evaluate the model's search efficiency. (Also available as `solver.benchmark`)

    This function solves a set of scrambles and reports on key performance metrics,
    providing a snapshot of the solver's efficiency under specific conditions.

    Args:
        solver: The solver instance to evaluate.
        num_samples (int): The number of scrambles to solve for the evaluation.
        beam_width (int): The beam width to use for the search.
        verbose (bool): Whether to display a progress bar.

    Returns:
        dict: A dictionary containing the mean results for solve time (`t`),
              solution length (`lmd`), and nodes expanded (`nodes`).
    """
    # Fail gracefully if dependencies are missing.
    if not _EVAL_DEPS_INSTALLED:
        raise ImportError(_EVAL_DEPS_ERROR_MSG)

    warnings.warn(
        "`evaluate_search_efficiency` is deprecated and has been renamed to `benchmark`. "
        "Please use `solver.benchmark()` instead. This alias will be removed in a future version.",
        DeprecationWarning,
    )
    dataset = get_dataset()

    results = pd.DataFrame(columns=["t", "lmd", "nodes"])

    for scramble in tqdm(
        dataset[: min(1000, num_samples)],
        desc=f"Solving... [model_id={solver.model_id}, {beam_width=}]",
        smoothing=False,
        disable=not verbose,
    ):
        res = solver(scramble, allow_wide=False, beam_width=beam_width)
        if res:
            results.loc[len(results)] = {
                "t": res["time"],
                "lmd": len(res["solutions"][0].split()),
                "nodes": res["num_nodes"],
            }

    return results.mean().to_dict()


benchmark = evaluate_search_efficiency


def evaluate_temporal_performance(
    solver,
    num_samples=1000,
    t_standard=1.0,
    beam_width_space=2 ** np.arange(6 if device.type == "cpu" else 10, 16 + 1),
    verbose=False,
):
    """
    Evaluate the model's performance on a downstream *temporal* performance.

    This function evaluates the model's performance by solving a set of scrambles
    using different beam widths. It then fits a predictor to model the relationship
    between solution length and time, and predicts the solution length at t=1.

    Args:
        solver: The solver to evaluate.
        num_samples (int): The number of samples to use for evaluation.
        t_standard (float): The standard time.
        beam_width_space (array): The beam widths to use for evaluation.
        verbose (bool): Whether to display a progress bar.

    Returns:
        float: The predicted solution length at t=1.
    """
    # Fail gracefully if dependencies are missing.
    if not _EVAL_DEPS_INSTALLED:
        raise ImportError(_EVAL_DEPS_ERROR_MSG)

    warnings.warn(
        "`evaluate_temporal_performance` is deprecated and will be removed in future versions. "
    )

    dataset = get_dataset()

    results = pd.DataFrame(columns=["beam_width", "t", "lmd"])

    for beam_width in beam_width_space.tolist():  # 16 ~ 16384
        for scramble in tqdm(
            dataset[:num_samples],
            desc=f"Solving... [model_id={solver.model_id}, {beam_width=}]",
            smoothing=False,
            disable=not verbose,
        ):
            res = solver(scramble, allow_wide=False, beam_width=beam_width)
            if res:
                results.loc[len(results)] = {
                    "beam_width": beam_width,
                    "t": res["time"],
                    "lmd": len(res["solutions"][0].split()),
                }
        if results[results["beam_width"] == beam_width]["t"].mean() > t_standard:
            break

    """Fit the lambda predictor"""
    results_agg = results.groupby("beam_width").agg({"t": "mean", "lmd": "mean"}).reset_index()
    t, lmd = results_agg.t.values, results_agg.lmd.values
    E = results.lmd.min()
    # linearize thoguh log-transformation
    t_log, lmd_log = np.log(t, dtype=np.double), np.log(lmd - E + 1e-9)

    # fit a predictor
    reg = LinearRegression().fit(
        t_log.reshape(-1, 1), lmd_log, t
    )  # Weight by time for better accuracy around t=1

    t_range = np.linspace(t.min(), max(t.max(), 1))

    lmd_range = E + np.exp(reg.predict(np.log(t_range).reshape(-1, 1)))
    # Pay special attention to \lambda(t=1)
    lmd_t1 = E + np.exp(reg.predict([[0]])).item()  # Predict at t=1 (log(1) = 0)

    """visualize results & esimates"""
    plt.figure(figsize=(4.8, 4.8))
    for beam_width in set(results.beam_width):
        results_by_beam_width = results[results["beam_width"] == beam_width]
        plt.scatter(
            results_by_beam_width.t, results_by_beam_width.lmd, label=f"{beam_width=}", alpha=0.2
        )
        plt.scatter(
            results_by_beam_width.t.mean(),
            results_by_beam_width.lmd.mean(),
            s=64,
            marker="+",
            c="k",
            alpha=0.5,
        )
    plt.semilogx(t_range, lmd_range, c="k", lw=1, label=rf"$\lambda'(t=1)={lmd_t1:.3f}$")
    plt.axvline(1, lw=1, ls=":", c="k")
    plt.scatter(1, lmd_t1, s=128, marker="x", c="k")
    plt.xlabel("Time (s)")
    plt.ylabel("Solution Length")
    plt.title(f"Temporal efficiency (N={num_samples})")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(cache_dir, "temporal_performance.png"))
    plt.show()

    return float(lmd_t1)
