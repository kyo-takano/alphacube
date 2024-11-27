"""
Module for evaluating the performance of a solver.

This module provides functions to evaluate the performance of a solver on a dataset.
It includes functions to evaluate the search efficiency and temporal performance of a solver.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from . import device, logargs, logger

cache_dir = os.path.expanduser("~/.cache/alphacube")


def get_dataset(filename="deepcubea-dataset--cube3.json"):
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
        import requests

        with requests.get(
            os.path.join("https://storage.googleapis.com/alphacube/", filename), stream=True
        ) as r:
            with open(filepath, "wb") as output:
                for chunk in r.iter_content(chunk_size=8192):
                    output.write(chunk)

    logger.info(f"[grey50]Saved to {filepath}", **logargs)
    try:
        with open(filepath) as f:
            dataset = json.load(f)
    except Exception as e:
        os.remove(filepath)
        raise ValueError(
            f"The model file appears to be broken, most likely because of permission error (deleted):\n{e}"
        )

    return dataset


def evaluate_search_efficiency(
    solver,
    num_samples=1000,
    beam_width=2**10 if device.type == "cpu" else 2**13,
    verbose=False,
):
    """
    Evaluate the model's performance on a downstream *temporal* performance.

    This function evaluates the model's performance by solving a set of scrambles
    using different beam widths. It then fits a predictor to model the relationship
    between solution length and time, and predicts the solution length achievable in just a second.

    Args:
        solver: The solver to evaluate.
        num_samples (int): The number of samples to use for evaluation.
        beam_width (int): The beam width to use for evaluation.
        verbose (bool): Whether to display a progress bar.

    Returns:
        dict: The mean results.
    """
    dataset = get_dataset()

    results = pd.DataFrame(columns=["t", "lmd", "nodes"])

    for scramble in tqdm(
        dataset[:num_samples],
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
