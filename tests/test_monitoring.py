import pandas as pd

from cooling_load.monitoring import population_stability_index


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    reference = pd.Series(range(1000))
    assert population_stability_index(reference, reference) == 0
    assert population_stability_index(reference, reference + 1000) > 0.25
