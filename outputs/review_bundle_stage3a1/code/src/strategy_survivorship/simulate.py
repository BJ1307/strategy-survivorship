"""Stage 1 data generating process and random-stream bookkeeping.

The benchmark DGP is deliberately the simplest thing that still poses the
question the supervisor asked: fixed volatility, iid Gaussian daily returns,

    r_t = S * sigma_ann / D  +  (sigma_ann / sqrt(D)) * eps_t ,   eps_t ~ N(0, 1).

``S`` is a *property of the generator*, not of any realised path.  Paths are never
recentred or rescaled after the fact to force their sample Sharpe to equal S --
doing so would destroy exactly the estimation noise the detectors must cope with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import STREAM_ORDER, Stage1Config


@dataclass(frozen=True)
class PathSet:
    """A block of simulated paths plus the ground truth kept away from detectors."""

    name: str
    returns: np.ndarray  # shape (n_paths, n_days)
    sharpe_true: float  # annualised Sharpe of the generator
    is_valid: bool  # ground-truth state label -- evaluation only
    stream: str  # which named random stream produced it

    @property
    def n_paths(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.returns.shape[1])


def make_streams(cfg: Stage1Config) -> dict[str, np.random.SeedSequence]:
    """Spawn one independent SeedSequence per named stream.

    ``SeedSequence.spawn`` is the documented NumPy way to obtain streams that are
    statistically independent *by construction* rather than by hoping that
    seed+1 is far enough away from seed.  The assignment order is pinned in
    ``config.STREAM_ORDER``.
    """
    root = np.random.SeedSequence(cfg.root_seed)
    children = root.spawn(len(STREAM_ORDER))
    return dict(zip(STREAM_ORDER, children, strict=True))


def stream_fingerprint(streams: dict[str, np.random.SeedSequence]) -> dict[str, dict]:
    """Serialisable description of each stream, for the run metadata."""
    return {
        name: {
            "entropy": int(ss.entropy),
            "spawn_key": [int(k) for k in ss.spawn_key],
            "n_children_spawned": int(ss.n_children_spawned),
            "pool_size": int(ss.pool_size),
        }
        for name, ss in streams.items()
    }


def simulate_returns(
    seed_seq: np.random.SeedSequence,
    n_paths: int,
    n_days: int,
    sharpe_annual: float,
    cfg: Stage1Config,
) -> np.ndarray:
    """Draw ``n_paths x n_days`` daily excess returns from the Stage 1 DGP."""
    rng = np.random.default_rng(seed_seq)
    eps = rng.standard_normal((n_paths, n_days))
    return cfg.daily_drift(sharpe_annual) + cfg.sigma_daily * eps


def build_pathsets(cfg: Stage1Config) -> dict[str, PathSet]:
    """Calibration set and the two independent test sets.

    All detectors are later scored on *these very arrays* -- no detector gets a
    private redraw.
    """
    streams = make_streams(cfg)
    H = cfg.horizon_days
    spec = [
        ("calibration_valid", cfg.n_calibration, cfg.sharpe_valid, True),
        ("test_valid", cfg.n_test_valid, cfg.sharpe_valid, True),
        ("test_invalid", cfg.n_test_invalid, cfg.sharpe_invalid, False),
    ]
    out: dict[str, PathSet] = {}
    for name, n, sharpe, is_valid in spec:
        out[name] = PathSet(
            name=name,
            returns=simulate_returns(streams[name], n, H, sharpe, cfg),
            sharpe_true=sharpe,
            is_valid=is_valid,
            stream=name,
        )
    return out


def simulate_diagnostic_paths(cfg: Stage1Config) -> dict[str, np.ndarray]:
    """One fixed *valid* path plus its two single-shock copies.

    The path index and the stream are fixed in the config before anything is
    plotted; the shock is a pure addition on one day, so the three series are
    identical everywhere else.
    """
    streams = make_streams(cfg)
    idx = cfg.diagnostic_path_index
    block = simulate_returns(
        streams["diagnostic"], idx + 1, cfg.horizon_days, cfg.sharpe_valid, cfg
    )
    base = block[idx].copy()
    day0 = cfg.shock_day - 1  # 1-based day -> 0-based index
    amp = cfg.shock_in_daily_sigma * cfg.sigma_daily

    up = base.copy()
    up[day0] += amp
    down = base.copy()
    down[day0] -= amp
    return {"base": base, "shock_plus": up, "shock_minus": down}
