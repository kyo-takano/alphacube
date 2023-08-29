"""
Beam Search Algorithm

This module provides a function to perform beam search and find solutions for a given state.

Function:
    ``beam_search``: Perform beam search to find solutions in a Rubik's Cube environment.
"""

import time
from rich import print
from . import logger
import torch
import numpy as np
from contextlib import nullcontext
import warnings; warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MAX_BATCH_SIZE = 2**16 # The maximum number of states processed by DNN at a time.

@torch.no_grad()
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

    # Ensure that the model is set to evaluation mode
    model.eval()

    # Initialize ergonomic bias if provided
    if ergonomic_bias is not None:
        # Zero-fill N/A and normalize to 1.
        ergonomic_bias = np.array([ ergonomic_bias.get(m, 0) for m in env.moves ], dtype=np.half).reshape(1, -1)
        if np.all(ergonomic_bias[:, 18:]==0):
            logger.info('All wide moves (e.g., r2, f2) seem to be 0 ── starting to solve only with flat moves...')
            # If wide moves are disabled, switch to flat-move mode
            env.allow_wide = False
            env.moves_ix_inference = env.moves_ix_inference[:18]
            ergonomic_bias = ergonomic_bias[:,:18]
            logger.debug(f"{ergonomic_bias.shape=}")
        ergonomic_bias /= ergonomic_bias.mean()
        logger.debug(f"{ergonomic_bias=}")
    else:
        # There is no point in wide moves with no ergonomic bias
        env.allow_wide = False
        env.moves_ix_inference = env.moves_ix_inference[:18]

    # Context: Use mixed precision training if GPU is available
    ctx = torch.autocast(DEVICE, dtype=torch.float16) if torch.cuda.is_available() else nullcontext()

    # Initialize candidate paths and their corresponding states and estimated probabilities
    candidates = {
        "cumprod": np.array([1.], dtype=np.half),   # Cumulative probability of candidate paths
        "state": env.state[None, :],                # Current state
        "path": np.array([[]], dtype=np.byte)       # Sequence of move indices constituting each path
    }
    solutions = {'solutions': [], 'num_nodes': 0, 'time': time.monotonic()}

    # Debug utilities

    def validate_state(i, centered=True):
        env.state[:] = candidates["state"][i]
        try:
            env.validate(centered=centered)
        except:
            print("[Path]")
            print(' '.join([env.moves[_] for _ in candidates["path"][i, :]]))
            print("[State]")
            env.show(flat=True)
            raise ValueError("State validation failed.")

    if logger.level <= 10:
        logger.debug(f"{env.moves=}")
        logger.debug("env.state:")
        env.show(flat=True)

    ## Execute ##

    for depth in range(max_depth):
        if logger.level <= 20:
            print(f"Current depth: {depth}", end="\r")

        ### Get a probability distribution for each candidate state ###

        with ctx:
            # Compute batch probabilities
            if len(candidates["cumprod"]) < MAX_BATCH_SIZE:
                batch_x = torch.from_numpy(candidates["state"]).to(DEVICE)
                batch_p = model(batch_x)
                batch_p = batch_p.detach().cpu().numpy()
            else:
                batch_p = np.concatenate([
                    model(torch.from_numpy(batch_split).to(DEVICE)).cpu().numpy()
                    for batch_split in np.array_split(candidates["state"], len(candidates["cumprod"]) // MAX_BATCH_SIZE + 1)
                ])
        batch_p = batch_p.astype(np.half)

        ### Evaluate each candidate base on cumprod ###

        # Apply ergonomic bias if given
        if ergonomic_bias is not None:
            # Expand batch_p -> include wide moves
            if env.allow_wide:
                batch_p = np.tile(batch_p, 2)
            batch_p = np.multiply(batch_p, ergonomic_bias)

        # Calculate log-sum of the cumulative probability of each candidate path
        # Equivalent to `candidates["cumprod"] = np.multiply(batch_p, candidates["cumprod"][:, None]).reshape(-1)`
        candidates["cumprod"] = (np.log(batch_p) + candidates["cumprod"][:, None]).reshape(-1)

        # Expand states & paths as the next-depth candidates
        candidates["state"] = np.repeat(candidates["state"], len(env.moves_ix_inference), axis=0)
        candidates["path"] = np.hstack((
            np.repeat(candidates["path"], len(env.moves_ix_inference), axis=0),
            np.tile(env.moves_ix_inference, batch_p.shape[0])[:, None]
        ))

        # Prune candidates based on previous moves
        if depth:
            # Face indices
            mod_first_last_moves = candidates["path"][:, -1] // 3
            mod_second_last_moves = candidates["path"][:, -2] // 3
            if env.allow_wide:
                # Reduce wide group as ordinary group
                mod_first_last_moves = mod_first_last_moves % 6
                mod_second_last_moves = mod_second_last_moves % 6

            # 1. Two subsequent moves on a same face
            prune_idx = mod_second_last_moves == mod_first_last_moves
            if depth > 1:
                # Two moves on a same face with an opposite move in between
                prune_idx = np.logical_or(prune_idx, np.logical_and(
                    candidates["path"][:, -3] // 3 == mod_first_last_moves,
                    mod_second_last_moves // 2 == mod_first_last_moves // 2
                ))
            candidates =  {k:v[~prune_idx] for k,v in candidates.items()}

        # Sort & select best k candidates
        sorted_indices = np.argsort(-candidates["cumprod"])[:beam_width]
        candidates =  {k:v[sorted_indices] for k,v in candidates.items()}

        ### Update states based on the expanded paths ###

        if not env.allow_wide:
            logger.debug("[ sticker replacement ]")
            state_ix = np.arange(0, len(sorted_indices), dtype=np.intc)[:, None] * 54 # [[0], [54], [108], [162], ...]
            move_indices = candidates["path"][:, -1]
            target_ix = state_ix + env.sticker_target_ix[move_indices]
            source_ix = state_ix + env.sticker_source_ix[move_indices]
            if logger.level <= 10:
                assert state_ix.ndim == 2 and move_indices.ndim == 1
                logger.debug(f"{state_ix.shape=}\n{move_indices.shape=}") # (8, 1)\n(8, )
                logger.debug(f"{target_ix.shape=}\n{source_ix.shape=}") # (8, 20)\n(8, 20) 
                for i in range(len(sorted_indices)):
                    validate_state(i)
            # Sticker replacement on the batch level (executed in the flattened view)
            candidates["state"].flat[target_ix.flatten()] = candidates["state"].flat[source_ix.flatten()]
            if logger.level <= 10:
                for i in range(len(sorted_indices)):
                    validate_state(i)
        else:
            # State transitions are split into two parts: flat moves and wide moves
            # 1. Apply ordinary moves
            indices_flat = np.argwhere(candidates["path"][:, -1] < 18)
            if len(indices_flat):
                logger.debug("[ Flat-move transition ]")
                move_indices = candidates["path"][indices_flat, -1].flatten()

                logger.debug(" - sticker replacement")
                state_ix = indices_flat * 54
                target_ix, source_ix = state_ix + env.sticker_target_ix[move_indices], state_ix + env.sticker_source_ix[move_indices]
                if logger.level <= 10:
                    assert state_ix.ndim == 2 and move_indices.ndim == 1
                    logger.debug(f"{state_ix.shape=}\n{move_indices.shape=}") # (8, 1)\n(8, )
                    logger.debug(f"{target_ix.shape=}\n{source_ix.shape=}") # (8, 20)\n(8, 20) 
                    for i in indices_flat.flatten():
                        validate_state(i)
                candidates["state"].flat[target_ix.flatten()] = candidates["state"].flat[source_ix.flatten()]
                if logger.level <= 10:
                    for i in indices_flat.flatten():
                        validate_state(i)

            # 2. Apply wide moves
            indices_wide = np.argwhere(candidates["path"][:, -1] > 17)
            if len(indices_wide):
                logger.debug("[ Wide-move transition ]")
                move_indices = candidates["path"][indices_wide, -1].flatten() - 18

                logger.debug(" - Sticker replacement")
                state_ix = indices_wide * 54 # array([[  54], [ 162], [ 270], [ 378], [ 486], [ 594], [ 648] ...
                target_ix = state_ix + env.sticker_target_ix_wide[move_indices]
                source_ix = state_ix + env.sticker_source_ix_wide[move_indices]
                if logger.level <= 10:
                    assert state_ix.ndim == 2 and move_indices.ndim == 1
                    logger.debug(f"{state_ix.shape=}\n{move_indices.shape=}") # (8, 1)\n(8, )
                    logger.debug(f"{target_ix.shape=}\n{source_ix.shape=}") # (8, 20)\n(8, 20) 
                    for i in indices_flat.flatten():
                        validate_state(i)
                candidates["state"].flat[target_ix.flatten()] = candidates["state"].flat[source_ix.flatten()] 
                if logger.level <= 10:
                    for i in indices_wide.flatten():
                        validate_state(i, centered=False)

                logger.debug(" - Reset color indices according to center colors after wide moves")
                # 1. Get center colors
                centers = candidates["state"][indices_wide, env.CENTER_INDICES]
                indices_wide = indices_wide.flatten() # can be flattened once sliced
                # 2. Sort center indices, to which current colors are re-indexed.
                mapping = np.argsort(centers, axis=1)
                mapping_indices = np.arange(len(indices_wide))[:, None]
                logger.debug(f"{mapping.shape=}\n{mapping_indices.shape=}\n{candidates['state'][indices_wide, :].shape=}")
                candidates["state"][indices_wide, :] = mapping[mapping_indices, candidates["state"][indices_wide, :]]
                if logger.level <= 10:
                    for i in indices_wide:
                        validate_state(i)

        ### Check if solved & done ###

        # Convert candidate states to bytes for goal collation AND uniqueness comparison
        candidates_state_bytes = np.array([bytes(c_state.tolist()) for c_state in candidates["state"]])
        is_solved = candidates_state_bytes == bytes(env.GOAL.tolist())
        # Add the number of current candidates to the node count
        solutions["num_nodes"] += len(sorted_indices)

        # Check for solved states
        solved_indices = np.where(is_solved)[0]
        for c_ix in solved_indices:
            path = [env.moves[i] for i in candidates["path"][c_ix]]
            if path[:-2] not in [p[:-2] for p in solutions['solutions']]:
                solutions['solutions'].append(path)

        # Return when solutions found or max_depth reached
        if solutions["solutions"] and depth+1 in [max_depth, len(solutions["solutions"][0]) + extra_depths]:
            # Finally include the time taken
            solutions['time'] = time.monotonic() - solutions['time']
            # Convert each list of solutions to string notation
            solutions["solutions"] = [" ".join(path) for path in solutions["solutions"]]
            if logger.level <= 20:
                print() # Avoid conflict with the current-depth log
            return solutions

        ### Otherwise, dedupe & pass to the next depth ###

        # Prune duplicated candidates based on their state uniqueness
        unique_indices = np.unique(candidates_state_bytes, return_index=True)[1]
        if len(solved_indices):
            unique_indices = np.setdiff1d(unique_indices, solved_indices)
        candidates = {k: v[unique_indices] for k, v in candidates.items()}
