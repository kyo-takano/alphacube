import pytest
import alphacube


BEAM_WIDTH_CPU = 256
BEAM_WIDTH_GPU = 1024


@pytest.fixture
def beam_width():
    """Return the beam width based on the device type."""
    return BEAM_WIDTH_CPU if alphacube.device.type == "cpu" else BEAM_WIDTH_GPU


@pytest.fixture
def num_samples():
    return 100
