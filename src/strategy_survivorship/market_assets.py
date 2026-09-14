"""Asset registry for the broadened cross-market baseline, by measurement category.

Three categories, kept apart on purpose. Mixing them is the mistake this module
exists to prevent.

``return``
    A return on a strategy, a factor or an investment vehicle. Enters the full
    fixed-model contrast.

``price_change``
    A change in a spot price, an index level or an FX rate. It also enters the
    contrast, but it is **not an investment return**: it contains no financing, no
    roll, no carry and no dividend. Every report says so where the number appears.

``market_state``
    A level series describing the state of a market -- an implied volatility, a
    yield. Levels and changes are reported and nothing else. A change in one of
    these is not a return on anything, and comparing it with the noise kernel would
    at most evaluate the CHANGE; it would not mean a level process had been
    generated, still less a strategy with a known Sharpe ratio.

Substitutions that are forbidden here and are not made: a Treasury yield is not a
swaption volatility, an FX spot change is not an FX carry total return, and GLD or
crude is not a commodity QIS.
"""

from __future__ import annotations

import datetime as dt

CATEGORIES = {
    "return": "a return on a strategy, factor or investment vehicle",
    "price_change": "a spot price, index level or FX rate change -- NOT an investment "
                    "return: no financing, no roll, no carry, no dividend",
    "market_state": "a level series describing market state; levels and changes only",
}

# key -> registry entry. `snapshot` is the file-name stem written by the acquisition
# step, so every object is bound to a specific archived file.
ASSETS: dict[str, dict] = {
    # ---- returns ----------------------------------------------------------
    "ff_momentum": {
        "label": "Momentum factor (daily)", "category": "return",
        "snapshot": "F-F_Momentum_Factor_daily", "reader": "kenfrench",
        "calendar": "nyse", "units": "daily factor return, decimal",
        "note": "a zero-investment long-short portfolio return, gross of costs",
    },
    "gld": {
        "label": "SPDR Gold Shares (GLD)", "category": "return",
        "snapshot": "GLD_HISTORICAL", "reader": "gld_xlsx", "calendar": "nyse",
        "units": "share closing price in USD; returns are decimals",
        "convention": "Closing Price only, never mixed with NAV",
        "note": "an investment vehicle. The sponsor fee is paid in gold, so ounces "
                "per share decline over time and a GLD return is NOT a pure gold "
                "price return. GLD pays no distributions, so no distribution "
                "adjustment is needed.",
    },
    # ---- price changes ----------------------------------------------------
    "sp500": {"label": "S&P 500", "category": "price_change", "snapshot": "SP500",
              "reader": "fred", "series_id": "SP500", "calendar": "nyse",
              "units": "index points, price index, dividends excluded"},
    "nasdaq100": {"label": "Nasdaq-100", "category": "price_change",
                  "snapshot": "NASDAQ100", "reader": "fred", "series_id": "NASDAQ100",
                  "calendar": "nyse",
                  "units": "index points, price index, dividends excluded"},
    "nikkei225": {"label": "Nikkei 225", "category": "price_change",
                  "snapshot": "NIKKEI225", "reader": "fred", "series_id": "NIKKEI225",
                  "calendar": "jpx",
                  "units": "index points (JPY), price index, dividends excluded"},
    "sx5e": {"label": "EURO STOXX 50", "category": "price_change",
             "snapshot": "STOXX_hbrbcpe", "reader": "stoxx", "column": 2,
             "calendar": "source_grid", "in_baseline": False,
             "units": "index points (EUR), price index, dividends excluded",
             "status": "source found and column confirmed, COVERAGE INSUFFICIENT",
             "note": "the official STOXX file hbrbcpe.txt is public and its own header "
                     "row names the columns SX5P | SX5E | SXXP | SXXE | ... so column "
                     "index 2 is EURO STOXX 50 -- confirmed by the file, not inferred. "
                     "But this archive ends on 2016-10-04 and does not reach the study "
                     "window, so the object is NOT in the baseline. The previous "
                     "round's conclusion that Euro Stoxx was unavailable was reached "
                     "from GUESSED FRED ids returning 404, which was the wrong kind of "
                     "inference; this is the right reason, and it is a coverage gap "
                     "rather than an absence of a source.",
             "what_is_needed": "the companion current-period STOXX file, or a licensed "
                               "feed, covering 2017-2025"},
    "brent": {"label": "Brent crude spot", "category": "price_change",
              "snapshot": "DCOILBRENTEU", "reader": "fred",
              "series_id": "DCOILBRENTEU", "calendar": "source_grid",
              "units": "USD per barrel, Europe Brent FOB spot",
              "note": "a SPOT price. Not a futures return and not a continuous "
                      "contract: no roll, no financing, no collateral yield."},
    "eurusd": {"label": "EUR/USD spot", "category": "price_change",
               "snapshot": "DEXUSEU", "reader": "fred", "series_id": "DEXUSEU",
               "calendar": "source_grid",
               "units": "US dollars per one euro (FRED DEXUSEU quotes USD per EUR), "
                        "so a rise means the euro appreciating",
               "note": "a SPOT rate change. It contains no interest differential, so "
                       "it is not an FX carry total return."},
    "usdjpy": {"label": "USD/JPY spot", "category": "price_change",
               "snapshot": "DEXJPUS", "reader": "fred", "series_id": "DEXJPUS",
               "calendar": "source_grid",
               "units": "Japanese yen per one US dollar (FRED DEXJPUS quotes JPY per "
                        "USD), so a rise means the dollar appreciating",
               "note": "a SPOT rate change; not an FX carry total return."},
    # ---- market state -----------------------------------------------------
    "vix": {"label": "VIX", "category": "market_state", "snapshot": "CBOE_VIX",
            "reader": "cboe", "column": "CLOSE",
            "units": "percent, annualised implied volatility; a change is in VIX "
                     "POINTS and is not a return"},
    "ovx": {"label": "OVX (crude oil implied vol)", "category": "market_state",
            "snapshot": "CBOE_OVX", "reader": "cboe", "column": "OVX",
            "units": "percent, annualised implied volatility; changes in points"},
    "gvz": {"label": "GVZ (gold implied vol)", "category": "market_state",
            "snapshot": "CBOE_GVZ", "reader": "cboe", "column": "GVZ",
            "units": "percent, annualised implied volatility; changes in points"},
    "dgs2": {"label": "US 2y Treasury yield", "category": "market_state",
             "snapshot": "DGS2", "reader": "fred", "series_id": "DGS2",
             "units": "percent per annum, a LEVEL; changes reported in basis points"},
    "dgs10": {"label": "US 10y Treasury yield", "category": "market_state",
              "snapshot": "DGS10", "reader": "fred", "series_id": "DGS10",
              "units": "percent per annum, a LEVEL; changes reported in basis points"},
    "dgs30": {"label": "US 30y Treasury yield", "category": "market_state",
              "snapshot": "DGS30", "reader": "fred", "series_id": "DGS30",
              "units": "percent per annum, a LEVEL; changes reported in basis points"},
    "wti": {"label": "WTI crude spot", "category": "market_state",
            "snapshot": "DCOILWTICO", "reader": "fred", "series_id": "DCOILWTICO",
            "units": "USD per barrel, Cushing OK spot",
            "returns_computable": False,
            "why_not": "the series prints -36.98 on 2020-04-20. A simple return across "
                       "a sign change is not a meaningful quantity, so no return "
                       "series is formed. The observation is NOT deleted, NOT "
                       "winsorised and NOT log-transformed; the level series is "
                       "reported instead and the negative print is shown."},
}

RETURN_LIKE = tuple(k for k, v in ASSETS.items()
                    if v["category"] in ("return", "price_change")
                    and v.get("returns_computable", True)
                    and v.get("in_baseline", True))
NOT_IN_BASELINE = tuple(k for k, v in ASSETS.items() if not v.get("in_baseline", True))
STATE_LIKE = tuple(k for k, v in ASSETS.items()
                   if v["category"] == "market_state"
                   or not v.get("returns_computable", True))

DESCRIBE = (dt.date(2017, 1, 1), dt.date(2021, 12, 31))
CONTRAST = (dt.date(2022, 1, 1), dt.date(2023, 12, 31))
