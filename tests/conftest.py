from dataclasses import replace
from pathlib import Path

import pytest

from strategy_survivorship.config import DEFAULT


@pytest.fixture(scope="session")
def cfg():
    return DEFAULT


@pytest.fixture(scope="session")
def small_cfg():
    """Same statistical definitions, small enough for a fast test run."""
    return replace(DEFAULT, n_calibration=600, n_test_valid=600, n_test_invalid=600)


def read_artifact_csv(path: str):
    """Read a generated CSV, or skip: generated artefacts are not under version control.

    Paths are relative to the working directory, as in the pipeline itself, so run pytest
    from the repository root after generating the artefact.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        pytest.skip(f"{p} is a generated artefact and is not present; run the stage that writes it first")
    return pd.read_csv(p)
