import pandas as pd
import deimos
import pytest

from tests import localfile


@pytest.fixture()
def ms1_peaks():
    ms1 = deimos.load(localfile("resources/isotope_example_data.h5"), key="ms1")
    return deimos.peakpick.persistent_homology(
        ms1, dims=["mz", "drift_time", "retention_time"], radius=[2, 10, 0]
    )


# need to test more configurations
def test_detect(ms1_peaks):
    isotopes = deimos.isotopes.detect(
        ms1_peaks,
        dims=["mz", "drift_time", "retention_time"],
        tol=[0.1, 0.2, 0.3],
        delta=1.003355,
        max_isotopes=5,
        max_charge=1,
        max_error=50e-6,
    )

    # grab the most intense isotopic pattern
    isotopes = isotopes.sort_values(by="intensity", ascending=False)
    isotopes = isotopes.iloc[0, :]

    assert abs(isotopes["mz"] - 387.024353) <= 1e-3
    assert isotopes["n"] == 4
    assert all([x <= 50e-6 for x in isotopes["error"]])
    assert isotopes["dx"] == [1.003355, 2.00671, 3.010065, 4.01342]


def test_detect_mz_only(ms1_peaks):
    """Test isotope detection with only m/z dimension (single spectrum mode)."""
    # Select features from a single retention time to simulate a single spectrum
    # Use the retention time with the most features
    rt_value = ms1_peaks.groupby("retention_time").size().idxmax()
    single_spectrum = ms1_peaks[ms1_peaks["retention_time"] == rt_value].copy()
    
    # Keep only mz and intensity columns (simulate a simple spectrum)
    single_spectrum = single_spectrum[["mz", "intensity"]].reset_index(drop=True)
    
    # Verify we have enough data
    assert len(single_spectrum) > 50  # Should have plenty of features
    
    # Run isotope detection with only mz dimension
    isotopes = deimos.isotopes.detect(
        single_spectrum,
        dims=["mz"],
        tol=[0.1],  # Only m/z tolerance needed
        delta=1.003355,
        max_isotopes=5,
        max_charge=1,
        max_error=50e-6,
    )
    
    # Should find some isotope patterns
    assert len(isotopes) > 0
    
    # Check that all patterns have valid m/z values
    assert all(isotopes["mz"] > 0)
    
    # Check that n (number of isotopes) is reasonable
    assert all(isotopes["n"] >= 1)
    assert all(isotopes["n"] <= 5)
    
    # Verify error values are within tolerance
    for errors in isotopes["error"]:
        assert all([e <= 50e-6 for e in errors])


def test_detect_unsorted_input():
    """Test that detect() handles unsorted (descending m/z) input correctly."""
    # Sorted ascending reference
    features_sorted = pd.DataFrame(
        {
            "mz": [557.975, 558.312, 558.644],
            "drift_time": [285.84, 285.97, 285.93],
            "intensity": [1870, 831232, 792192],
        }
    )

    # Same features but in reverse m/z order
    features_unsorted = features_sorted.iloc[::-1].reset_index(drop=True)

    kwargs = dict(
        dims=["mz", "drift_time"],
        tol=[0.02, 1.0],
        max_charge=3,
        max_isotopes=1,
        max_error=50e-6,
        require_lower_intensity=False,
    )

    result_sorted = deimos.isotopes.detect(features_sorted, **kwargs)
    result_unsorted = deimos.isotopes.detect(features_unsorted, **kwargs)

    # Both orderings must yield the same number of isotopic patterns
    assert len(result_sorted) == len(result_unsorted)
    # Patterns found should be consistent (non-empty)
    assert len(result_sorted) > 0
