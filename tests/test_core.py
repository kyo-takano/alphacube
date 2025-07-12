import pytest
import alphacube


@pytest.mark.core
@pytest.mark.main
def test_solve_basic(beam_width):
    """
    https://alphacube.dev/docs/getting-started/index.html#basic
    """
    # Load a trained DNN
    alphacube.load()  # the default model

    # Solve the cube using a given scramble
    result = alphacube.solve(
        scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
        beam_width=beam_width,
    )
    assert result is not None


@pytest.mark.core
@pytest.mark.main
def test_solve_wider(beam_width):
    """
    https://alphacube.dev/docs/getting-started/index.html#improving-quality
    """
    alphacube.load()
    result = alphacube.solve(
        scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
        beam_width=beam_width * 4,
    )

    assert result is not None


@pytest.mark.core
@pytest.mark.advanced
def test_solve_extra(beam_width):
    """
    https://alphacube.dev/docs/getting-started/index.html#allowing-a-few-extra-moves
    """
    alphacube.load()
    result = alphacube.solve(
        scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
        beam_width=beam_width,
        extra_depths=1,
    )
    assert result is not None


@pytest.mark.core
@pytest.mark.advanced
def test_solve_with_bias(beam_width):
    """
    https://alphacube.dev/docs/getting-started/index.html#applying-ergonomic-bias
    """
    alphacube.load()
    ergonomic_bias = {
        "U": 0.9,
        "U'": 0.9,
        "U2": 0.8,
        "R": 0.8,
        "R'": 0.8,
        "R2": 0.75,
        "L": 0.55,
        "L'": 0.4,
        "L2": 0.3,
        "F": 0.7,
        "F'": 0.6,
        "F2": 0.6,
        "D": 0.3,
        "D'": 0.3,
        "D2": 0.2,
        "B": 0.05,
        "B'": 0.05,
        "B2": 0.01,
        "u": 0.45,
        "u'": 0.45,
        "u2": 0.4,
        "r": 0.3,
        "r'": 0.3,
        "r2": 0.25,
        "l": 0.2,
        "l'": 0.2,
        "l2": 0.15,
        "f": 0.35,
        "f'": 0.3,
        "f2": 0.25,
        "d": 0.15,
        "d'": 0.15,
        "d2": 0.1,
        "b": 0.03,
        "b'": 0.03,
        "b2": 0.01,
    }

    result = alphacube.solve(
        scramble="D U F2 L2 U' B2 F2 D L2 U R' F' D R' F' U L D' F' D R2",
        beam_width=beam_width * 4,
        ergonomic_bias=ergonomic_bias,
    )
    assert result is not None
