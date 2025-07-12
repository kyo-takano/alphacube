import pytest
import numpy as np
from alphacube.env import Cube3


@pytest.mark.env
def test_reset_axes():
    """Test that reset_axes correctly re-orients the cube after a wide move."""
    env = Cube3(allow_wide=True)
    env.finger("u")  # A wide move 'u' swaps the E-slice, changing centers.

    # Centers are no longer [0, 1, 2, 3, 4, 5]
    assert not np.array_equal(env.state[env.CENTER_INDICES], env.CENTERS_HAT)

    env.reset_axes()

    # After reset_axes, the centers should be correct again.
    # The cube is not solved, but the frame of reference is fixed.
    np.testing.assert_array_equal(env.state[env.CENTER_INDICES], env.CENTERS_HAT)
