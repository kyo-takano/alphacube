import pytest
import numpy as np
from alphacube.search import _get_prune_idx


@pytest.mark.search
def test_pruning_same_face_subsequent_moves():
    """Test pruning of paths like '... R R2'."""
    # Paths: [..., U, U'], [..., R, R2], [..., L, F] (don't prune)
    paths = np.array(
        [
            [0, 1],  # U, U' (U is face 0, U' is face 0) -> Prune
            [7, 8],  # R', R2 (R is face 3, R2 is face 3) -> Prune
            [12, 9],  # F, D (F is face 5, D is face 1) -> Keep
        ]
    )

    prune_mask = _get_prune_idx(paths, allow_wide=False, depth=1)
    np.testing.assert_array_equal(prune_mask, [True, True, False])


@pytest.mark.search
def test_pruning_sandwich_moves():
    """Test pruning of paths like '... U D U2'."""
    # Paths:
    # [U, D, U2] -> faces are 0, 1, 0. D is opposite of U. -> Prune
    # [R, F, R'] -> faces are 3, 5, 3. F is not opposite of R. -> Keep
    # [L, R, L'] -> faces are 2, 3, 2. R is opposite of L. -> Prune
    paths = np.array(
        [
            [0, 3, 2],  # U, D, U2
            [6, 15, 7],  # R, F, R'
            [9, 6, 10],  # L, R, L'
        ]
    )

    prune_mask = _get_prune_idx(paths, allow_wide=False, depth=2)
    np.testing.assert_array_equal(prune_mask, [True, False, True])
