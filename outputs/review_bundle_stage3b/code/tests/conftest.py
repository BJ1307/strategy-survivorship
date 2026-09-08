from dataclasses import replace

import pytest

from strategy_survivorship.config import DEFAULT


@pytest.fixture(scope="session")
def cfg():
    return DEFAULT


@pytest.fixture(scope="session")
def small_cfg():
    """Same statistical definitions, small enough for a fast test run."""
    return replace(DEFAULT, n_calibration=600, n_test_valid=600, n_test_invalid=600)
