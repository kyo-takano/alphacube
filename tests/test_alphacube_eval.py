import pytest
import alphacube


@pytest.mark.eval
def test_evaluate_search_efficiency(num_samples, beam_width):
    alphacube.load()
    result = alphacube.solver.evaluate_search_efficiency(num_samples, beam_width, verbose=True)
    assert result["lmd"] < 30


@pytest.mark.eval
def test_evaluate_temporal_performance(num_samples):
    alphacube.load()
    result = alphacube.solver.evaluate_temporal_performance(num_samples, verbose=True)
    assert result < 30
