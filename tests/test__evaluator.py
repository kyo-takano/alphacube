import pytest
import alphacube


@pytest.mark.eval
def test_benchmark(num_samples, beam_width):
    alphacube.load()
    result = alphacube.solver.benchmark(num_samples, beam_width, verbose=True)
    assert result["lmd"] < 30


@pytest.mark.skip("Deprecated feature")
@pytest.mark.eval
def test_evaluate_temporal_performance(num_samples):
    alphacube.load()
    result = alphacube.solver.evaluate_temporal_performance(num_samples, verbose=True)
    assert result < 30
