"""
Rubik's Cube Environment

This module defines the Cube3 class representing a 3x3x3 Rubik's Cube in the Half-Turn Metric.

Class:
    `Cube3`: A class for 3x3x3 Rubik's Cube in Half-Turn Metric.
"""

import os
import random

import numpy as np
import rich
import torch


class Cube3:
    """
    A class for 3x3x3 Rubik's Cube in Half-Turn Metric (HTM).

    This class provides methods to manipulate and solve a 3x3x3 Rubik's Cube using the half-turn metric.
    It defines the cube's initial and goal states, available moves, and methods for cube manipulation.

    **Representation**:
    > **Order of faces**:
    > ```
    >    0
    >  2 5 3 4
    >    1
    > ```
    >
    > **Order of stickers on each face**:
    > ```
    >   2   5   8
    >   1   4   7
    >  [0]  3   6
    > ```
    >
    > **Indices of state** (each starting with `9 * (n-1)`):
    > ```
    >                 2   5   8
    >                 1   4   7
    >                [0]  3   6
    >   20  23  26   47  50  53  29  32  35  38  41  44
    >   19  22  25   46  49  52  28  31  34  37  40  43
    >  [18] 21  24  [45] 48  51 [27] 30  33 [36] 39  42
    >                11  14  17
    >                10  13  16
    >                [9] 12  15
    > ```
    >
    > **Colors** (`indices // 9`):
    > ```
    >                 0   0   0
    >                 0   0   0
    >                 0   0   0
    >    2   2   2    5   5   5   3   3   3   4   4   4
    >    2   2   2    5   5   5   3   3   3   4   4   4
    >    2   2   2    5   5   5   3   3   3   4   4   4
    >                 1   1   1
    >                 1   1   1
    >                 1   1   1
    > ```

    Attributes:
        state (ndarray): Current cube state represented as an array of sticker colors.
        GOAL (ndarray): Fixed goal state represented as an array of sticker colors.
        moves (list): List of possible cube moves (face and direction).
        allow_wide (bool): Flag indicating whether wide moves are allowed.
        max_depth (int): The maximum scramble depth for the data generator.
        sticker_target (dict): A dictionary mapping move strings to lists of **target** sticker indices.
        sticker_source (dict): A dictionary mapping move strings to lists of **source** sticker indices.
        sticker_target_ix (ndarray): A 2D numpy array mapping move indices to **target** sticker indices for normal moves.
        sticker_source_ix (ndarray): A 2D numpy array mapping move indices to **source** sticker indices for normal moves.
        sticker_target_ix_wide (ndarray): A 2D numpy array mapping move indices to **target** sticker indices for wide moves.
        sticker_source_ix_wide (ndarray): A 2D numpy array mapping move indices to **source** sticker indices for wide moves.
    """

    def __init__(self, allow_wide=True, max_depth=20):
        self.allow_wide = allow_wide

        # Define initial and goal state
        self.reset()
        self.GOAL = np.arange(0, 6 * 9, dtype=np.int64) // 9

        # Define moves
        faces = ["U", "D", "L", "R", "B", "F"]
        if self.allow_wide:
            faces += ["d", "u", "r", "l", "f", "b"]  # Essentially the same as above
        degrees = ["", "'", "2"]
        degrees_inference = ["'", "", "2"]  # inverse

        # List of moves in HTM notation (e.g., R', U, F2)
        self.moves = [f + n for f in faces for n in degrees]
        # List mapping the indices of predicted last *training* moves to their inverse for inference.
        # e.g., 0 -> 1, 1 -> 0, 2 -> 2
        self.moves_ix_inference = [
            self.moves.index(f + n) for f in faces for n in degrees_inference
        ]

        # Vectorize the sticker group replacement operations
        self.__vectorize_moves()

        # Utilities for wide move postprocessing
        self.CENTER_INDICES = [4, 13, 22, 31, 40, 49]
        self.CENTERS_HAT = np.arange(0, 6, dtype=np.int64)

        """ For potential training with `__iter__` """
        self.max_depth = max_depth
        self.moves_ix = [self.moves.index(f + n) for f in faces for n in degrees]

    def show(self, flat=False, palette=["white", "yellow", "orange1", "red", "blue", "green"]):
        """
        Display the cube's current state.

        Args:
            flat (bool): Whether to display the state in flat form.
            palette (list): List of colors for representing stickers.
        """
        palette = ["white", "black", "blue", "red", "pink1", "green"]
        state_by_face = self.state.reshape(6, 9)
        if not flat:
            state_by_face = state_by_face[:, [2, 5, 8, 1, 4, 7, 0, 3, 6]].reshape(6, 3, 3)
        state_by_face = str(state_by_face)
        for i, color in zip(range(6), palette):
            state_by_face = state_by_face.replace(str(i), f"[{color}]{i}[/{color}]")
        rich.print(state_by_face)
        rich.print()

    def validate(self, state=None, centered=True):
        """
        Validate the cube's state and arrangement.

        Args:
            centered (bool): Whether centers should be centered or not.

        Raises:
            ValueError: If the cube's state or arrangement is invalid.
        """
        if state is not None:
            self.state[:] = state

        centers = self.state[self.CENTER_INDICES]

        if centered and not np.all(centers == self.CENTERS_HAT):
            # Must be [0, 1, 2, 3, 4, 5]
            raise ValueError("Invalid cube state: Centers are not in their solved positions.")
        if not centered and np.all(centers == self.CENTERS_HAT):
            # Must NOT be [0, 1, 2, 3, 4, 5]
            raise ValueError(
                "Invalid cube state: Expected a non-centered state, but centers are in solved positions."
            )

        if not np.all(np.sort(centers) == self.CENTERS_HAT):
            # Must be [0, 1, 2, 3, 4, 5] when sorted
            self.show(flat=True)
            raise ValueError(
                "Invalid cube state: The set of center colors is incorrect. Expected one of each color index (0-5)."
            )
        elif not np.all(np.sort(self.state) == self.GOAL):
            rich.print(np.sort(self.state).reshape(6, 9), "!= env.goal")
            raise ValueError("Inconsistent number of colors.")

    def reset(self):
        """Resets the cube state to the solved state."""
        self.state = np.arange(0, 6 * 9, dtype=np.int64) // 9

    def reset_axes(self):
        """
        Reset color indices according to the given center colors.
        Useful when fat moves are applied or when an unexpected perspective is specified.
        """
        centers = self.state[self.CENTER_INDICES]
        if not np.all(centers == self.CENTERS_HAT):
            # Sort center indices, to which current colors are re-indexed.
            mapping = np.argsort(centers)
            assert mapping.shape == (6,)
            self.state = mapping[self.state]
            assert self.state.shape == (54,)

    def is_solved(self):
        """Checks if the cube is in the solved state."""
        return np.all(self.state == self.GOAL)

    def finger(self, move):
        """
        Apply a single move on the cube state using move string.

        Args:
            move (str): Move string in HTM notation.
        """
        self.state[self.sticker_target[move]] = self.state[self.sticker_source[move]]

    def finger_ix(self, ix):
        """
        Apply a single move using its index for faster execution than `.finger`.
        Checks if the move index corresponds to a normal move (ix < 18) or a wide move and applies
        the state change using pre-calculated index arrays.

        Args:
            ix (int): Index of the move to apply.
        """
        if ix < 18:
            self.state[self.sticker_target_ix[ix]] = self.state[self.sticker_source_ix[ix]]
        else:
            self.state[self.sticker_target_ix_wide[ix % 18]] = self.state[
                self.sticker_source_ix_wide[ix % 18]
            ]

    def apply_scramble(self, scramble):
        """
        Applies a sequence of moves (scramble) to the cube state.

        Args:
            scramble (str or list): Sequence of moves in HTM notation or list.
        """
        if isinstance(scramble, str):
            scramble = scramble.split()
        for m in scramble:
            self.finger(m)

    def __iter__(self):
        """
        Create an infinite generator of scrambled states and solution sequences.

        This method is intended for model training. On each iteration, it generates
        a new random scramble of `max_depth` moves, avoiding trivial move sequences.
        It yields the history of states and the corresponding move that led to each state.

        Yields:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X (np.ndarray): A (max_depth, 54) array of cube states.
                - y (np.ndarray): A (max_depth,) array of move indices that generated the states.
        """
        assert not self.allow_wide
        while True:
            X = np.zeros((self.max_depth, 54), dtype=int)
            y = []

            self.reset()
            seq = []
            for _ in range(self.max_depth):
                if seq:
                    while True:
                        ix = random.choice(self.moves_ix)
                        # Skip subsequent moves on a same face
                        if ix // 3 == seq[-1] // 3:
                            continue
                        # Skip two moves on a same face with an opposite move in between
                        if _ > 1 and (ix // 3 == seq[-2] // 3 and ix // 6 == seq[-1] // 6):
                            continue
                        break
                else:
                    ix = random.choice(self.moves_ix)
                self.finger_ix(ix)
                seq.append(ix)
                # yield self.state, ix
                X[_, :] = self.state
                y.append(ix)
            yield X, np.array(y, dtype=int)

    def __vectorize_moves(self):
        """
        Vectorizes the sticker group replacement operations for faster computation.
        This method defines `self.sticker_target` and `self.sticker_source` to manage sticker colors (target is replaced by source).
        They define indices of target and source stickers so that the moves can be vectorized.
        """
        self.sticker_target, self.sticker_source = dict(), dict()

        # fmt: off
        self.sticker_replacement = {
            # Sticker A is replaced by another sticker at index B -> {A: B}
            "U": {
                0: 6, 1: 3, 2: 0, 3: 7, 5: 1, 6: 8, 7: 5, 8: 2, 20: 47, 23: 50, 26: 53, 29: 38, 32: 41, 35: 44, 38: 20, 41: 23, 44: 26, 47: 29, 50: 32, 53: 35,
            },
            "D": {9: 15, 10: 12, 11: 9, 12: 16, 14: 10, 15: 17, 16: 14, 17: 11, 18: 36, 21: 39, 24: 42, 27: 45, 30: 48, 33: 51, 36: 27, 39: 30, 42: 33, 45: 18, 48: 21, 51: 24,
            },
            "L": {
                0: 44, 1: 43, 2: 42, 9: 45, 10: 46, 11: 47, 18: 24, 19: 21, 20: 18, 21: 25, 23: 19, 24: 26, 25: 23, 26: 20, 42: 11, 43: 10, 44: 9, 45: 0, 46: 1, 47: 2,
            },
            "R": {
                6: 51, 7: 52, 8: 53, 15: 38, 16: 37, 17: 36, 27: 33, 28: 30, 29: 27, 30: 34, 32: 28, 33: 35, 34: 32, 35: 29, 36: 8, 37: 7, 38: 6, 51: 15, 52: 16, 53: 17,
            },
            "B": {
                2: 35, 5: 34, 8: 33, 9: 20, 12: 19, 15: 18, 18: 2, 19: 5, 20: 8, 33: 9, 34: 12, 35: 15, 36: 42, 37: 39, 38: 36, 39: 43, 41: 37, 42: 44, 43: 41, 44: 38,
            },
            "F": {
                0: 24, 3: 25, 6: 26, 11: 27, 14: 28, 17: 29, 24: 17, 25: 14, 26: 11, 27: 6, 28: 3, 29: 0, 45: 51, 46: 48, 47: 45, 48: 52, 50: 46, 51: 53, 52: 50, 53: 47,
            },
        }
        if self.allow_wide:
            # Slice moves
            self.sticker_replacement.update(
                {
                    # Definition: https://jperm.net/3x3/moves
                    "M": {
                        49: 4, 4: 40, 40: 13, 13: 49, 50: 5, 5: 39, 39: 14, 14: 50, 48: 3, 3: 41, 41: 12, 12: 48,
                    },
                    "S": {
                        31: 4, 4: 22, 22: 13, 13: 31, 32: 1, 1: 21, 21: 16, 16: 32, 30: 7, 7: 23, 23: 10, 10: 30,
                    },
                    "E": {
                        49: 22, 22: 40, 40: 31, 31: 49, 46: 19, 19: 37, 37: 28, 28: 46, 52: 25, 25: 43, 43: 34, 34: 52,
                    },
                }
            )
            # fmt: on
            # Wide moves
            self.sticker_replacement.update(
                {
                    "u": {
                        **self.sticker_replacement["U"],
                        **{v: k for k, v in self.sticker_replacement["E"].items()},
                    },
                    "d": {**self.sticker_replacement["D"], **self.sticker_replacement["E"]},
                    "l": {**self.sticker_replacement["L"], **self.sticker_replacement["M"]},
                    "r": {
                        **self.sticker_replacement["R"],
                        **{v: k for k, v in self.sticker_replacement["M"].items()},
                    },
                    "b": {
                        **self.sticker_replacement["B"],
                        **{v: k for k, v in self.sticker_replacement["S"].items()},
                    },
                    "f": {**self.sticker_replacement["F"], **self.sticker_replacement["S"]},
                }
            )

        for m in self.moves:
            if len(m) == 1:
                assert m in self.sticker_replacement
            else:
                if "'" in m:
                    self.sticker_replacement[m] = {
                        v: k for k, v in self.sticker_replacement[m[0]].items()
                    }
                elif "2" in m:
                    self.sticker_replacement[m] = {
                        k: self.sticker_replacement[m[0]][v]
                        for k, v in self.sticker_replacement[m[0]].items()
                    }
                else:
                    raise

            self.sticker_target[m] = list(self.sticker_replacement[m].keys())
            self.sticker_source[m] = list(self.sticker_replacement[m].values())

            for i, idx in enumerate(self.sticker_target[m]):
                assert self.sticker_replacement[m][idx] == self.sticker_source[m][i]

        # For index slicing
        # Normal moves
        self.sticker_target_ix = np.array(
            [np.array(self.sticker_target[m]) for m in self.moves[:18]]
        )
        self.sticker_source_ix = np.array(
            [np.array(self.sticker_source[m]) for m in self.moves[:18]]
        )
        # Wide moves
        if self.allow_wide:
            self.sticker_target_ix_wide = np.array(
                [np.array(self.sticker_target[m]) for m in self.moves[18:]]
            )
            self.sticker_source_ix_wide = np.array(
                [np.array(self.sticker_source[m]) for m in self.moves[18:]]
            )


class Dataset(torch.utils.data.Dataset):
    """
    Pseudo dataset class to infinitely yield random scrambles.

    > ```python title="Example"
    > batch_size = 1024
    > dl = get_dataloader(batch_size)
    > for i, (batch_x, batch_y) in zip(range(1000), dl):
    >     batch_x, batch_y = batch_x.to(device), batch_y.device().reshape(-1)
    > ```
    """

    def __init__(self, max_depth=20, num_workers=os.cpu_count()):
        self.num_workers = num_workers
        self.generators = [
            iter(Cube3(allow_wide=False, max_depth=max_depth)) for _ in range(num_workers)
        ]

    def __len__(self):
        return 0x7FFFFFFF  # max int possible with a single precision

    def __getitem__(self, i):
        return next(self.generators[i % self.num_workers])


def get_dataloader(
    batch_size,
    num_workers=min(os.cpu_count(), 32),  # DataLoader slightly slows down beyond 32 CPU cores
    max_depth=20,
    **dl_kwargs,
):
    """
    Create a DataLoader instance for generating random Rubik's Cube scrambles.

    Args:
        batch_size (int): The number of samples per batch.
        num_workers (int, optional): The number of worker processes to use for data loading.
            Defaults to the number of CPU cores or 32 (beyond which the return will diminish), whichever is smaller.
        max_depth (int, optional): The maximum depth of the scrambles. Defaults to 20.
        **dl_kwargs: Additional keyword arguments to pass to the DataLoader constructor.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance that yields batches of random scrambles.
    """
    ds = Dataset(max_depth=max_depth, num_workers=num_workers)
    return torch.utils.data.DataLoader(
        ds, num_workers=num_workers, batch_size=batch_size, **dl_kwargs
    )
