"""
This module provides a function to perform beam search and find solutions for a given state.

Function:
    `beam_search`: Perform beam search to find solutions in a Rubik's Cube environment.

"""

import time
import numpy as np
import torch

from .utils import logger, device

MAX_BATCH_SIZE = 2**16  # The maximum number of states forward-pass through a DNN at a time.


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
        env (Cube3): The Rubik's Cube environment representing the scrambled state.
        model (torch.nn.Module): DNN used to predict the probability distribution of next moves for every state.
        beam_width (int): The maximum number of candidates to keep at each step of the search.
        ergonomic_bias (dict or None): A dictionary specifying ergonomic bias for moves, if available.
        extra_depths (int): The number of additional depths to search beyond the first solution's depth.
        max_depth (int): The maximum depth to search, should be equal to  or greater than God's Number (20 for Rubik's Cube in HTM).

    Returns:
        dict | None:
            With at least one solution, a dictionary with the following keys:
                1. `"solutions"`: A list of optimal or near-optimal solutions found during the search.
                1. `"num_nodes"`: The total number of nodes expanded during the search.
                1. `"time"`: The time taken (in seconds) to complete the search.

            Otherwise, `None`.
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

    if logger.level <= 10:
        logger.debug(f"{env.moves=}")
        logger.debug("env.state:")
        env.show(flat=True)

    ## Execute ##
    for depth in range(max_depth):
        # Get a probability distribution for each candidate state
        batch_x = candidates["state"]
        batch_logprob = predict(model, batch_x, ergonomic_bias, env)
        candidates = update_candidates(candidates, batch_logprob, env, depth, beam_width)

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
            return solutions

        # Otherwise, dedupe & pass to the next depth
        # Prune duplicated candidates based on their state uniqueness
        unique_indices = np.unique(candidates_state_bytes, return_index=True)[1]
        if len(solved_indices):
            unique_indices = np.setdiff1d(unique_indices, solved_indices)
        candidates = {k: v[unique_indices] for k, v in candidates.items()}


def _reflect_setup(ergonomic_bias, env):
    """
    Initialize ergonomic bias if provided.

    Args:
        ergonomic_bias (dict or None): A dictionary specifying ergonomic bias for moves, if available.
        env (Cube3): The Rubik's Cube environment representing the scrambled state.

    Returns:
        ergonomic_bias (numpy.ndarray): The ergonomic bias for moves, if available.
        env (Cube3): The Rubik's Cube environment representing the scrambled state.
    """
    if ergonomic_bias is not None:
        # Zero-fill N/A and normalize to 1.
        ergonomic_bias = np.array(
            [ergonomic_bias.get(m, 0) for m in env.moves], dtype=np.single
        ).reshape(1, -1)
        if np.all(ergonomic_bias[:, 18:] == 0):
            logger.info(
                "Ergonomic bias for all wide moves is zero. "
                "Disabling wide moves and using a standard-move search space."
            )
            # If wide moves are disabled, switch to flat-move mode
            env.allow_wide = False
            env.moves_ix_inference = env.moves_ix_inference[:18]
            ergonomic_bias = ergonomic_bias[:, :18]
            logger.debug(f"{ergonomic_bias.shape=}")
        ergonomic_bias /= ergonomic_bias.mean()
        ergonomic_bias = np.log(ergonomic_bias)
        logger.debug(f"{ergonomic_bias=}")
    else:
        # There is no point in wide moves with no ergonomic bias
        env.allow_wide = False
        env.moves_ix_inference = env.moves_ix_inference[:18]

    return ergonomic_bias, env


@torch.inference_mode()
def predict(model, batch_x, ergonomic_bias, env):
    """
    Predict the probability distribution of next moves for every state.

    Args:
        model (torch.nn.Module): DNN used to predict the probability distribution of next moves for every state.
        batch_x (numpy.ndarray): Batch of states.
        ergonomic_bias (dict or None): A dictionary specifying ergonomic bias for moves, if available.
        env (Cube3): The Rubik's Cube environment representing the scrambled state.

    Returns:
        batch_logprob (numpy.ndarray): The log probability of each move for each state.

    :::note

    Inference with [Automatic Mixed Prevision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) is slightly faster than
    the simple half-precision (with `model.half()`) for some reasons.

    :::
    """
    batch_x = torch.from_numpy(batch_x).to(device)
    # with ctx:
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
    batch_logprob = logits.log_softmax(-1).detach().cpu().numpy()  # float32

    # Apply ergonomic bias if given
    if ergonomic_bias is not None:
        if env.allow_wide:
            batch_logprob = np.tile(batch_logprob, 2)
        # batch_p = np.multiply(batch_p, ergonomic_bias)
        batch_logprob += ergonomic_bias

    return batch_logprob


def update_candidates(candidates, batch_logprob, env, depth, beam_width):
    """
    Expand candidate paths with the predicted probabilities of next moves.

    Args:
        candidates (dict): A dictionary containing candidate paths, cumulative probabilities, and states.
        batch_logprob (numpy.ndarray): The log probability of each move for each state.
        env (Cube3): The Rubik's Cube environment representing the scrambled state.
        depth (int): The current depth of the search.
        beam_width (int): The maximum number of candidates to keep at each step of the search.

    Returns:
        candidates (dict): The updated dictionary containing candidate paths, cumulative probabilities, and states.
    """
    # Accumulate the log-probability of each candidate path
    # Non-log equivalent:
    # `candidates["cumprob"] = np.multiply(batch_logprob, candidates["cumprob"][:, None]).reshape(-1)`
    # with np.errstate(divide="ignore"):
    candidates["cumlogprob"] = (candidates["cumlogprob"][:, None] + batch_logprob).reshape(-1)

    # Expand states & paths as the next-depth candidates
    candidates["state"] = np.repeat(candidates["state"], len(env.moves_ix_inference), axis=0)
    candidates["path"] = np.hstack(
        (
            np.repeat(candidates["path"], len(env.moves_ix_inference), axis=0),
            np.tile(env.moves_ix_inference, batch_logprob.shape[0])[:, None],
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
    """
    Get the indices of candidates to prune based on previous moves.

    Args:
        candidates_paths (numpy.ndarray): The paths of candidate states.
        allow_wide (bool): Whether to allow wide moves.
        depth (int): The current depth of the search.

    Returns:
        prune_idx (numpy.ndarray): The indices of candidates to prune.

    :::note

    Using `numba.jit` actually *slows down* this function.

    :::
    """
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
    """
    Update states based on the expanded paths.

    Args:
        candidate_states (numpy.ndarray): The states of candidate states.
        candidate_paths (numpy.ndarray): The paths of candidate states.
        env (Cube3): The Rubik's Cube environment representing the scrambled state.

    Returns:
        candidate_states (numpy.ndarray): The updated states of candidate states.
    """
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
    """
    Perform sticker replacement on the batch level.

    Args:
        candidate_states (numpy.ndarray): The states of candidate states.
        target_ix (numpy.ndarray): The target indices for sticker replacement.
        source_ix (numpy.ndarray): The source indices for sticker replacement.

    Returns:
        candidate_states (numpy.ndarray): The updated states of candidate states.

    :::note

    Using `numba.jit` actually *slows down* this function.

    :::
    """
    # Sticker replacement on the batch level (executed in the flattened view)
    flat_states = candidate_states.ravel()
    flat_states[target_ix.flatten()] = flat_states[source_ix.flatten()]
    candidate_states = flat_states.reshape(candidate_states.shape)
    # Slightly faster than:
    # candidate_states.flat[target_ix.flatten()] = candidate_states.flat[source_ix.flatten()]
    # which also doesn't work with `numba.jit`
    return candidate_states
