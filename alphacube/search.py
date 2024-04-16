"""
Beam Search Algorithm

This module provides a function to perform beam search and find solutions for a given state.

Function:
    ``beam_search``: Perform beam search to find solutions in a Rubik's Cube environment.

Note:
- `numba.jit` *slows down* operations like `_get_prune_idx` and `_map_state`
"""

import time
import numpy as np
import torch
from rich import print as rprint
from contextlib import nullcontext

from . import device, dtype, logger

MAX_BATCH_SIZE = 2**16  # The maximum number of states forward-pass through a DNN at a time.


# Context: Use mixed precision training if GPU is available
ctx = torch.autocast(device.type, dtype) if device.type != "cpu" else nullcontext()


@torch.no_grad()
def predict(model, batch_x, beam_width, ergonomic_bias, env):
    batch_x = torch.from_numpy(batch_x).to(device)
    with ctx:
        if batch_x.shape[0] < MAX_BATCH_SIZE:
            logits = model(batch_x)
        else:
            logits = torch.cat(
                [
                    model(split)
                    for split in torch.tensor_split(
                        batch_x,
                        batch_x.shape[0] // MAX_BATCH_SIZE + 1,
                    )
                ]
            )
    batch_p = logits.softmax(-1) if beam_width > 1 else logits  # You can argmax from logits
    batch_p = batch_p.detach().cpu().numpy()  # float32

    # Apply ergonomic bias if given
    if ergonomic_bias is not None:
        if env.allow_wide:
            batch_p = np.tile(batch_p, 2)
        batch_p = np.multiply(batch_p, ergonomic_bias)

    return batch_p


def beam_search(
    env,
    model,
    beam_width,
    ergonomic_bias=None,
    extra_depths=0,
    max_depth=100,
):
    """
    Performs beam search to find solutions for a given scrambled state.

    Args:
        env (Cube3):
            The Rubik's Cube environment representing the scrambled state.

        model (torch.nn.Module):
            DNN used to predict the probability distribution of next moves for every state.

        beam_width (int):
            The maximum number of candidates to keep at each step of the search.

        ergonomic_bias (dict or None):
            A dictionary specifying ergonomic bias for moves, if available.

        extra_depths (int):
            The number of additional depths to search beyond the first solution's depth.

        max_depth (int):
            The maximum depth to search, should be equal to  or greater than God's Number (20 for Rubik's Cube in HTM).

    Returns:
        dict or None: A dictionary with the following keys:
            - 'solutions': A list of optimal or near-optimal solutions found during the search.
            - 'num_nodes': The total number of nodes expanded during the search.
            - 'time': The time taken (in seconds) to complete the search.

            If no solutions are found, returns None.

    """

    ## Setup ##
    model.eval()
    ergonomic_bias, env = _reflect_setup(ergonomic_bias, env)

    # Initialize candidate paths and their corresponding states and estimated probabilities
    candidates = dict(
        # Sequences of move indices constituting each path
        path=np.array([[]], dtype=np.byte),
        # Cumulative probability of candidate paths
        cumlogprob=np.array([0], dtype=np.single),
        # Corresponding states
        state=env.state[None, :],
    )
    solutions = dict(solutions=[], num_nodes=0, time=time.monotonic())

    # Debug utilities

    if logger.level < 20:
        logger.debug(f"{env.moves=}")
        logger.debug("env.state:")
        env.show(flat=True)

    ## Execute ##
    for depth in range(max_depth):
        if logger.level <= 20:
            rprint(f"Current depth: {depth}", end="\r")

        # Get a probability distribution for each candidate state
        batch_x = candidates["state"]
        batch_p = predict(model, batch_x, beam_width, ergonomic_bias, env)
        candidates = update_candidates(candidates, batch_p, env, depth, beam_width)

        # For record, add the number of current candidates to the node count
        solutions["num_nodes"] += candidates["path"].shape[0]

        ## Check if solved any & also done ##
        # Convert candidate states to bytes for goal collation;
        # the variable `candidates_state_bytes` is also gonna be used to dedupe candidates at the bottom
        candidates_state_bytes = np.array(
            [bytes(c_state.tolist()) for c_state in candidates["state"]]
        )

        # Check for solved states; stringify & append to the list of solutions
        solved_indices = np.where(candidates_state_bytes == bytes(env.GOAL.tolist()))[0]
        for c_ix in solved_indices:
            path = [env.moves[i] for i in candidates["path"][c_ix]]
            if path[:-2] not in [p[:-2] for p in solutions["solutions"]]:
                solutions["solutions"].append(path)

        # Return when solutions found or max_depth reached
        if solutions["solutions"] and depth + 1 in [
            max_depth,
            len(solutions["solutions"][0]) + extra_depths,
        ]:
            # Finally include the time taken
            solutions["time"] = time.monotonic() - solutions["time"]
            # Convert each list of solutions to string notation
            solutions["solutions"] = [" ".join(path) for path in solutions["solutions"]]
            if logger.level <= 20:
                rprint()  # To avoid conflict with the current-depth log
            return solutions

        # Otherwise, dedupe & pass to the next depth
        # Prune duplicated candidates based on their state uniqueness
        unique_indices = np.unique(candidates_state_bytes, return_index=True)[1]
        if len(solved_indices):
            unique_indices = np.setdiff1d(unique_indices, solved_indices)
        candidates = {k: v[unique_indices] for k, v in candidates.items()}


def _reflect_setup(ergonomic_bias, env):
    # Initialize ergonomic bias if provided
    if ergonomic_bias is not None:
        # Zero-fill N/A and normalize to 1.
        ergonomic_bias = np.array(
            [ergonomic_bias.get(m, 0) for m in env.moves], dtype=np.single
        ).reshape(1, -1)
        if np.all(ergonomic_bias[:, 18:] == 0):
            logger.info(
                "All wide moves (e.g., r2, f2) seem to be 0 ── starting to solve only with flat moves..."
            )
            # If wide moves are disabled, switch to flat-move mode
            env.allow_wide = False
            env.moves_ix_inference = env.moves_ix_inference[:18]
            ergonomic_bias = ergonomic_bias[:, :18]
            logger.debug(f"{ergonomic_bias.shape=}")
        ergonomic_bias /= ergonomic_bias.mean()
        logger.debug(f"{ergonomic_bias=}")
    else:
        # There is no point in wide moves with no ergonomic bias
        env.allow_wide = False
        env.moves_ix_inference = env.moves_ix_inference[:18]

    return ergonomic_bias, env


def update_candidates(candidates, batch_p, env, depth, beam_width):
    # Accumulate the log-probability of each candidate path
    # Non-log equivalent:
    # `candidates["cumprob"] = np.multiply(batch_p, candidates["cumprob"][:, None]).reshape(-1)`
    with np.errstate(divide="ignore"):
        candidates["cumlogprob"] = (candidates["cumlogprob"][:, None] + np.log(batch_p)).reshape(-1)

    # Expand states & paths as the next-depth candidates
    candidates["state"] = np.repeat(candidates["state"], len(env.moves_ix_inference), axis=0)
    candidates["path"] = np.hstack(
        (
            np.repeat(candidates["path"], len(env.moves_ix_inference), axis=0),
            np.tile(env.moves_ix_inference, batch_p.shape[0])[:, None],
        )
    )

    # Prune candidates based on previous moves
    if depth:
        prune_idx = _get_prune_idx(candidates["path"], env.allow_wide, depth)
        candidates = {k: v[~prune_idx] for k, v in candidates.items()}

    # Sort & select best k candidates
    sorted_indices = np.argsort(-candidates["cumlogprob"])[:beam_width]
    candidates = {k: v[sorted_indices] for k, v in candidates.items()}

    # Update states based on the expanded paths ###
    candidates["state"] = _update_states(candidates["state"], candidates["path"], env)
    return candidates


def _get_prune_idx(candidates_paths, allow_wide, depth):
    # Face indices
    mod_first_last_moves = candidates_paths[:, -1] // 3
    mod_second_last_moves = candidates_paths[:, -2] // 3
    if allow_wide:
        # Reduce wide group as ordinary group
        mod_first_last_moves = mod_first_last_moves % 6
        mod_second_last_moves = mod_second_last_moves % 6

    # 1. Two subsequent moves on a same face
    prune_idx = mod_second_last_moves == mod_first_last_moves
    if depth > 1:
        # Two moves on a same face with an opposite move in between
        prune_idx = np.logical_or(
            prune_idx,
            np.logical_and(
                candidates_paths[:, -3] // 3 == mod_first_last_moves,
                mod_second_last_moves // 2 == mod_first_last_moves // 2,
            ),
        )
    return prune_idx


def _update_states(candidate_states, candidate_paths, env):
    if not env.allow_wide:
        logger.debug("[ sticker replacement ]")
        state_ix = (
            np.arange(0, candidate_paths.shape[0], dtype=np.intc)[:, None] * 54
        )  # [[0], [54], [108], [162], ...]
        move_indices = candidate_paths[:, -1]
        target_ix = state_ix + env.sticker_target_ix[move_indices]
        source_ix = state_ix + env.sticker_source_ix[move_indices]
        if logger.level <= 10:
            assert state_ix.ndim == 2 and move_indices.ndim == 1
            logger.debug(f"{state_ix.shape=}\n{move_indices.shape=}")  # (8, 1)\n(8, )
            logger.debug(f"{target_ix.shape=}\n{source_ix.shape=}")  # (8, 20)\n(8, 20)
            for i in range(candidate_paths.shape[0]):
                env.validate(state=candidate_states[i], centered=True)
        candidate_states = _map_state(candidate_states, target_ix, source_ix)
        if logger.level <= 10:
            for i in range(candidate_paths.shape[0]):
                env.validate(state=candidate_states[i], centered=True)
    else:
        indices_flat = np.argwhere(candidate_paths[:, -1] < 18)
        if len(indices_flat):
            logger.debug("[ Flat-move transition ]")
            move_indices = candidate_paths[indices_flat, -1].flatten()
            logger.debug(" - sticker replacement")
            state_ix = indices_flat * 54
            target_ix, source_ix = (
                state_ix + env.sticker_target_ix[move_indices],
                state_ix + env.sticker_source_ix[move_indices],
            )
            candidate_states = _map_state(candidate_states, target_ix, source_ix)

        indices_wide = np.argwhere(candidate_paths[:, -1] > 17)
        if len(indices_wide):
            logger.debug("[ Wide-move transition ]")
            move_indices = candidate_paths[indices_wide, -1].flatten() - 18
            logger.debug(" - Sticker replacement")
            state_ix = indices_wide * 54
            target_ix = state_ix + env.sticker_target_ix_wide[move_indices]
            source_ix = state_ix + env.sticker_source_ix_wide[move_indices]

            candidate_states = _map_state(candidate_states, target_ix, source_ix)

            logger.debug(" - Reset color indices according to center colors after wide moves")
            centers = candidate_states[indices_wide, env.CENTER_INDICES]
            indices_wide = indices_wide.flatten()
            mapping = np.argsort(centers, axis=1)
            mapping_indices = np.arange(len(indices_wide))[:, None]
            logger.debug(
                f"{mapping.shape=}\n{mapping_indices.shape=}\n{candidate_states[indices_wide, :].shape=}"
            )
            candidate_states[indices_wide, :] = mapping[
                mapping_indices, candidate_states[indices_wide, :]
            ]
    return candidate_states


def _map_state(candidate_states, target_ix, source_ix):
    # Sticker replacement on the batch level (executed in the flattened view)
    flat_states = candidate_states.ravel()
    flat_states[target_ix.flatten()] = flat_states[source_ix.flatten()]
    candidate_states = flat_states.reshape(candidate_states.shape)
    # Slightly faster than:
    # candidate_states.flat[target_ix.flatten()] = candidate_states.flat[source_ix.flatten()]
    # which also doesn't work with `numba.jit`
    return candidate_states
