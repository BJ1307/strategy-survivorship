"""Real-data reference, stage 1: fetch, snapshot and clean a daily index series.

The protocol this implements is ``docs/MARKET_STAGE1_PROTOCOL.md`` and was
written before any diagnostic was produced.

What the series IS
------------------
FRED ``SP500`` is the S&P 500 daily closing **price** index: it excludes
dividends.  It is used here only to study the statistical shape of daily return
noise.  It is not a strategy return series, not a total-return or excess-return
series, and nothing here estimates or implies a Sharpe ratio for it.  The
project's own ``r_t = mu_S + sigma_0 eps_t`` (see ``config.py``) is a different
quantity and the two are never mixed.

Redistribution
--------------
FRED states that the S&P 500 data comes from S&P Dow Jones Indices and may not be
redistributed.  Raw snapshots and every per-day derived series stay under
``data/`` and ``outputs/market/``, both of which are kept out of Git.  The
downloader, the data dictionary, the provenance record and aggregate statistics
are the deliverables.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
USER_AGENT = "strategy-survivorship/0.1 (research; contact via repository)"

SOURCES: dict[str, dict] = {
    "sp500": {
        "role": "primary",
        "provider": "Federal Reserve Bank of St. Louis (FRED); index by S&P Dow Jones Indices LLC",
        "series_id": "SP500",
        "series_page": "https://fred.stlouisfed.org/series/SP500",
        "download_url": FRED_CSV.format(series_id="SP500"),
        "export_fields": "observation_date, SP500",
        "frequency": "daily, business days (Mon-Fri), blank on US market holidays",
        "units": "index level, points (S&P 500 PRICE index; dividends excluded)",
        "redistribution": "S&P Dow Jones Indices data; FRED states it may not be "
                          "redistributed. Raw and per-day derived files stay local.",
    },
    "nasdaq100": {
        "role": "cross-market, price index",
        "provider": "Federal Reserve Bank of St. Louis (FRED); index by Nasdaq, Inc.",
        "series_id": "NASDAQ100",
        "series_page": "https://fred.stlouisfed.org/series/NASDAQ100",
        "download_url": FRED_CSV.format(series_id="NASDAQ100"),
        "export_fields": "observation_date, NASDAQ100",
        "frequency": "daily, business days, blank on US market holidays",
        "units": "index level, points (PRICE index; dividends excluded)",
        "calendar": "nyse",
        "redistribution": "Nasdaq index data via FRED; kept local with the rest.",
    },
    "nikkei225": {
        "role": "cross-market, price index, different exchange calendar",
        "provider": "Federal Reserve Bank of St. Louis (FRED); index by Nikkei Inc.",
        "series_id": "NIKKEI225",
        "series_page": "https://fred.stlouisfed.org/series/NIKKEI225",
        "download_url": FRED_CSV.format(series_id="NIKKEI225"),
        "export_fields": "observation_date, NIKKEI225",
        "frequency": "daily, business days, blank on Japanese market holidays",
        "units": "index level, points (PRICE index; dividends excluded)",
        "calendar": "jpx",
        "calendar_note": "an encoded JPX rule set now provides the expected trading "
                         "days INDEPENDENTLY of the observed dates: New Year and year "
                         "end, the fixed and nth-weekday national holidays, the listed "
                         "equinoxes, the 2019 abdication and enthronement days "
                         "(2019-04-30, 05-01, 05-02, 10-22) and the 2020-2021 Olympic "
                         "moves. It is a rule set written here, not an authoritative "
                         "exchange calendar, so it is cross-checked against the source "
                         "file's blanks and the disagreements are reported. Over "
                         "2017-2023 the two agree on all 114 weekday closures, "
                         "including the 2019 Golden Week.",
        "redistribution": "Nikkei index data via FRED; kept local with the rest.",
    },
    "ff_momentum": {
        "role": "cross-market, a long-short FACTOR RETURN rather than a price index",
        "provider": "Kenneth R. French data library (Tuck School of Business)",
        "series_id": "F-F_Momentum_Factor_daily",
        "series_page": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                       "data_library.html",
        "download_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                        "F-F_Momentum_Factor_daily_CSV.zip",
        "export_fields": "YYYYMMDD, Mom",
        "frequency": "daily, US trading days",
        "units": "daily factor RETURN in PERCENT; 0.55 means 0.55%, i.e. 0.0055. "
                 "Converted to a decimal once on load and never again.",
        "calendar": "nyse",
        "definition": "MOM = average return of the two high-prior-return "
                      "value-weight portfolios minus the average of the two "
                      "low-prior-return portfolios, prior return measured from day "
                      "-250 to -21. A zero-investment long-short portfolio return, "
                      "NOT a price index and NOT net of any cost.",
        "missing_codes": "-99.99 and -999",
        "redistribution": "free for research use under the library's terms; kept "
                          "local with the rest and not redistributed here.",
    },
    "vix": {
        "role": "auxiliary, optional; failure here does not block the S&P 500 work",
        "provider": "Federal Reserve Bank of St. Louis (FRED); index by Cboe",
        "series_id": "VIXCLS",
        "series_page": "https://fred.stlouisfed.org/series/VIXCLS",
        "download_url": FRED_CSV.format(series_id="VIXCLS"),
        "export_fields": "observation_date, VIXCLS",
        "frequency": "daily, business days (Mon-Fri), blank on US market holidays",
        "units": "annualised implied volatility in PERCENT: a level of 20 means "
                 "about 20% annualised implied volatility, NOT a 20% return. It "
                 "looks forward about 30 calendar days, while RV_21 looks back "
                 "over 21 trading days, so the two are not expected to agree "
                 "point by point.",
        "redistribution": "Cboe index data via FRED; kept local with the rest.",
    },
}


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #


class FetchError(RuntimeError):
    """The bytes could not be obtained, or are not the data file we asked for."""


def _validate_fred_csv(payload: bytes, series_id: str) -> None:
    """Reject an error page, a login wall or an HTML redirect dressed as CSV."""
    if len(payload) < 64:
        raise FetchError(f"response is only {len(payload)} bytes; not a data file")
    head = payload[:4096].lstrip()
    if head[:1] == b"<" or b"<html" in head[:200].lower():
        raise FetchError("response looks like HTML, not CSV")
    first_line = payload.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    cols = [c.strip() for c in first_line.split(",")]
    if len(cols) != 2 or series_id not in cols:
        raise FetchError(f"unexpected CSV header {first_line!r}; expected a "
                         f"date column and {series_id!r}")


def fetch_bytes(url: str, timeout: float = 30.0, retries: int = 3,
                backoff: float = 2.0, sleep=time.sleep) -> bytes:
    """GET with a timeout and a FINITE number of retries.

    Retries only transient failures (timeout, connection reset, 5xx, 429).  A
    404 or a 403 is not retried: the resource is not going to appear.
    """
    if retries < 1:
        raise ValueError("retries must be at least 1")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise FetchError(f"HTTP {resp.status}")
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < retries:
            sleep(backoff * attempt)
    raise FetchError(f"failed after {retries} attempts: {last}")


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_snapshot(key: str, raw_dir: Path, timeout: float = 30.0,
                      retries: int = 3, now: dt.datetime | None = None) -> dict:
    """Fetch one source, write the RAW bytes untouched, and record provenance.

    The snapshot is the file of record: FRED's ``SP500`` only keeps a rolling
    ten-year window, so a later download will not contain the earliest dates of
    this study.
    """
    src = SOURCES[key]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    payload = fetch_bytes(src["download_url"], timeout=timeout, retries=retries)
    _validate_fred_csv(payload, src["series_id"])
    path = raw_dir / f"{src['series_id']}_{stamp}.csv"
    path.write_bytes(payload)
    frame = read_fred_csv(path, src["series_id"])
    obs = frame["value"].notna()
    rec = {
        "key": key,
        **{k: v for k, v in src.items()},
        "downloaded_at_utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
        "raw_file": str(path),
        "raw_bytes": len(payload),
        "raw_sha256": sha256_of(payload),
        "rows_in_file": int(len(frame)),
        "rows_with_a_value": int(obs.sum()),
        "rows_blank": int((~obs).sum()),
        "date_range_in_file": [str(frame["date"].min().date()),
                               str(frame["date"].max().date())],
    }
    (raw_dir / f"{src['series_id']}_{stamp}.provenance.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def download_kenfrench_zip(key: str, raw_dir: Path, timeout: float = 45.0,
                           retries: int = 3, now: dt.datetime | None = None) -> dict:
    """Fetch a Kenneth French zip, keep the raw bytes, record provenance."""
    import zipfile
    src = SOURCES[key]
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    payload = fetch_bytes(src["download_url"], timeout=timeout, retries=retries)
    if payload[:2] != b"PK":
        raise FetchError("response is not a zip archive")
    path = raw_dir / f"{src['series_id']}_{stamp}.zip"
    path.write_bytes(payload)
    with zipfile.ZipFile(path) as z:
        members = z.namelist()
    frame = read_kenfrench_daily(path)
    rec = {"key": key, **{k: v for k, v in src.items()},
           "downloaded_at_utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
           "raw_file": str(path), "raw_bytes": len(payload),
           "raw_sha256": sha256_of(payload), "zip_members": members,
           "rows_in_file": int(len(frame)),
           "rows_with_a_value": int(frame["value"].notna().sum()),
           "rows_blank": int(frame["value"].isna().sum()),
           "date_range_in_file": [str(frame["date"].min().date()),
                                  str(frame["date"].max().date())]}
    (raw_dir / f"{src['series_id']}_{stamp}.provenance.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def read_kenfrench_daily(path: Path) -> pd.DataFrame:
    """Two columns, ``date`` and ``value``, with the PERCENT units made explicit.

    The file carries a prose preamble, a header row, ``YYYYMMDD`` dates, values in
    PERCENT, missing codes ``-99.99`` / ``-999``, and a trailing copyright line.
    The percent-to-decimal conversion happens HERE and nowhere else.
    """
    import zipfile
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            text = z.read(z.namelist()[0]).decode("utf-8", "replace")
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or len(parts[0]) != 8 or not parts[0].isdigit():
            continue
        try:
            v = float(parts[1])
        except ValueError:
            continue
        rows.append((parts[0], np.nan if v in (-99.99, -999.0) else v / 100.0))
    if not rows:
        raise ValueError(f"{path}: no daily rows recognised")
    out = pd.DataFrame(rows, columns=["date", "value"])
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    return out.sort_values("date", ignore_index=True)


def read_xlsx_sheet(path: Path, sheet_index: int = 0) -> list[list]:
    """Minimal .xlsx reader built on the standard library only.

    Written rather than adding openpyxl so the pinned environment in
    ``requirements-lock.txt`` is not changed for one file format. It handles what
    these vendor exports actually use: inline and shared strings, numeric cells and
    the 1900 date serial. It is not a general Excel implementation.
    """
    import re
    import xml.etree.ElementTree as ET
    import zipfile
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{ns}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
        sheets = sorted(n for n in z.namelist()
                        if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if sheet_index >= len(sheets):
            raise IndexError(f"{path} has {len(sheets)} sheets")
        root = ET.fromstring(z.read(sheets[sheet_index]))
        rows = []
        for row in root.iter(f"{ns}row"):
            cells: dict[int, object] = {}
            for c in row.findall(f"{ns}c"):
                ref = c.get("r", "")
                col = 0
                for ch in ref:
                    if ch.isalpha():
                        col = col * 26 + (ord(ch.upper()) - 64)
                    else:
                        break
                v = c.find(f"{ns}v")
                if c.get("t") == "s" and v is not None:
                    val = shared[int(v.text)]
                elif c.get("t") == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter(f"{ns}t"))
                elif v is not None:
                    try:
                        val = float(v.text)
                    except (TypeError, ValueError):
                        val = v.text
                else:
                    val = None
                cells[col - 1] = val
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(i) for i in range(width)])
    return rows


def excel_serial_to_date(serial: float) -> dt.date:
    """Excel's 1900 serial, with its deliberate 1900-02-29 off-by-one."""
    return dt.date(1899, 12, 30) + dt.timedelta(days=int(serial))


def read_fred_csv(path: Path, series_id: str) -> pd.DataFrame:
    """Two tidy columns: ``date`` (datetime64) and ``value`` (float, NaN if blank).

    FRED writes one row per weekday and leaves the value blank on a US market
    holiday; older exports use ``.``.  Both become NaN, and the blank ROW is kept,
    because its presence is what marks the day as a non-trading weekday.
    """
    frame = pd.read_csv(path, na_values=["", "."], keep_default_na=True)
    cols = list(frame.columns)
    if len(cols) != 2 or series_id not in cols:
        raise ValueError(f"{path}: unexpected columns {cols}")
    date_col = [c for c in cols if c != series_id][0]
    out = pd.DataFrame({
        "date": pd.to_datetime(frame[date_col], errors="raise"),
        "value": pd.to_numeric(frame[series_id], errors="coerce"),
    })
    return out.sort_values("date", ignore_index=True)


# --------------------------------------------------------------------------- #
# trading calendar
# --------------------------------------------------------------------------- #


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian computus.  Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _observed(day: dt.date) -> dt.date:
    """NYSE rule: a Saturday holiday is taken on Friday, a Sunday one on Monday."""
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    nxt = dt.date(year + (month == 12), (month % 12) + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


# Unscheduled full-day closures inside the study window.  These are one-off
# events, not rules, so they are listed rather than derived.
AD_HOC_CLOSURES: tuple[tuple[str, str], ...] = (
    ("2018-12-05", "National day of mourning, George H. W. Bush"),
    ("2025-01-09", "National day of mourning, Jimmy Carter"),
)


def nyse_holidays(year: int) -> dict[dt.date, str]:
    """Scheduled NYSE full-day closures for one year, as {date: name}.

    Encoded as rules and tested, because no market-calendar package is in the
    pinned environment.  Juneteenth became an exchange holiday in 2022.
    """
    out: dict[dt.date, str] = {}

    def put(day: dt.date, name: str) -> None:
        out[_observed(day)] = name

    put(dt.date(year, 1, 1), "New Year's Day")
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[easter_sunday(year) - dt.timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        put(dt.date(year, 6, 19), "Juneteenth National Independence Day")
    put(dt.date(year, 7, 4), "Independence Day")
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    put(dt.date(year, 12, 25), "Christmas Day")
    # The year filter also implements the NYSE's one asymmetry: a New Year's Day
    # falling on a Saturday is NOT taken on the preceding Friday, so the date
    # _observed() produced lands in the previous year and drops out here.
    return {d: n for d, n in out.items() if d.year == year and d.weekday() < 5}


def market_holidays(start: dt.date, end: dt.date) -> dict[dt.date, str]:
    out: dict[dt.date, str] = {}
    for year in range(start.year - 1, end.year + 2):
        out.update(nyse_holidays(year))
    for iso, name in AD_HOC_CLOSURES:
        out[dt.date.fromisoformat(iso)] = name
    return {d: n for d, n in out.items() if start <= d <= end}


# --------------------------------------------------------------------------- #
# Japan Exchange Group calendar
# --------------------------------------------------------------------------- #

# The equinox holidays are astronomical and are listed rather than derived.
JPX_EQUINOXES: dict[int, tuple[str, str]] = {
    2017: ("2017-03-20", "2017-09-23"), 2018: ("2018-03-21", "2018-09-23"),
    2019: ("2019-03-21", "2019-09-23"), 2020: ("2020-03-20", "2020-09-22"),
    2021: ("2021-03-20", "2021-09-23"), 2022: ("2022-03-21", "2022-09-23"),
    2023: ("2023-03-21", "2023-09-23"),
}
# One-off national holidays and the Olympic-year moves, listed because they are
# events rather than rules.
JPX_SPECIAL: dict[int, tuple[tuple[str, str], ...]] = {
    2019: (("2019-04-30", "Abdication"), ("2019-05-01", "Enthronement"),
           ("2019-05-02", "Bridge holiday"),
           ("2019-10-22", "Enthronement ceremony")),
    2020: (("2020-07-23", "Marine Day (moved)"), ("2020-07-24", "Sports Day (moved)"),
           ("2020-08-10", "Mountain Day (moved)")),
    2021: (("2021-07-22", "Marine Day (moved)"), ("2021-07-23", "Sports Day (moved)"),
           ("2021-08-09", "Mountain Day (observed)")),
}
JPX_MOVED_AWAY: dict[int, tuple[str, ...]] = {
    2020: ("marine", "sports", "mountain"),
    2021: ("marine", "sports", "mountain"),
}


def jpx_holidays(year: int) -> dict[dt.date, str]:
    """Best-effort JPX closure rules for one year in 2017-2023.

    This is a RULE SET WRITTEN HERE, not an authoritative exchange calendar. It is
    always cross-checked against the source file's own blank rows and the
    disagreements are reported rather than resolved in its favour.
    """
    if not 2017 <= year <= 2023:
        raise ValueError(f"the encoded JPX rules cover 2017-2023 only, not {year}")
    out: dict[dt.date, str] = {}

    def sub(day: dt.date, name: str) -> None:
        """A holiday falling on a Sunday is observed on the next free weekday."""
        d = day
        while d.weekday() == 6 or d in out:
            d += dt.timedelta(days=1)
        out[d] = name

    for d in (1, 2, 3):
        out[dt.date(year, 1, d)] = "New Year (exchange closed)"
    out[dt.date(year, 12, 31)] = "Year end (exchange closed)"
    out[_nth_weekday(year, 1, 0, 2)] = "Coming of Age Day"
    sub(dt.date(year, 2, 11), "National Foundation Day")
    if year >= 2020:
        sub(dt.date(year, 2, 23), "Emperor's Birthday")
    if year <= 2018:
        sub(dt.date(year, 12, 23), "Emperor's Birthday")
    ve, ae = JPX_EQUINOXES[year]
    sub(dt.date.fromisoformat(ve), "Vernal Equinox")
    sub(dt.date.fromisoformat(ae), "Autumnal Equinox")
    sub(dt.date(year, 4, 29), "Showa Day")
    sub(dt.date(year, 5, 3), "Constitution Memorial Day")
    sub(dt.date(year, 5, 4), "Greenery Day")
    sub(dt.date(year, 5, 5), "Children's Day")
    moved = JPX_MOVED_AWAY.get(year, ())
    if "marine" not in moved:
        out[_nth_weekday(year, 7, 0, 3)] = "Marine Day"
    if "mountain" not in moved:
        sub(dt.date(year, 8, 11), "Mountain Day")
    out[_nth_weekday(year, 9, 0, 3)] = "Respect for the Aged Day"
    if "sports" not in moved:
        out[_nth_weekday(year, 10, 0, 2)] = "Sports Day"
    sub(dt.date(year, 11, 3), "Culture Day")
    sub(dt.date(year, 11, 23), "Labour Thanksgiving Day")
    for iso, name in JPX_SPECIAL.get(year, ()):
        out[dt.date.fromisoformat(iso)] = name
    return {d: n for d, n in out.items() if d.year == year and d.weekday() < 5}


def jpx_market_holidays(start: dt.date, end: dt.date) -> dict[dt.date, str]:
    out: dict[dt.date, str] = {}
    for year in range(start.year, end.year + 1):
        out.update(jpx_holidays(year))
    return {d: n for d, n in out.items() if start <= d <= end}


CALENDARS = {"nyse": market_holidays, "jpx": jpx_market_holidays}

#: Years each encoded rule set is willing to answer for.  ``None`` means the rules are
#: generated per year and carry no hard bound; a pair is inclusive.  Callers that need
#: a day just outside a window -- a previous close, say -- must consult this instead of
#: calling the calendar and catching the exception, so that "not checked" is recorded
#: as a fact rather than swallowed.
CALENDAR_YEARS: dict[str, tuple[int, int] | None] = {"nyse": None, "jpx": (2017, 2023)}


def calendar_covers(calendar: str, day: dt.date) -> bool:
    """Whether `calendar`'s encoded rules can speak about `day`."""
    if calendar not in CALENDARS:
        return False
    span = CALENDAR_YEARS.get(calendar)
    return True if span is None else span[0] <= day.year <= span[1]


def expected_trading_days_for(calendar: str, start: dt.date,
                              end: dt.date) -> list[dt.date]:
    """Weekdays in [start, end] that the named calendar says were open."""
    if calendar not in CALENDARS:
        raise ValueError(f"unknown calendar {calendar!r}")
    holidays = CALENDARS[calendar](start, end)
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur not in holidays:
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def expected_trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Weekdays in [start, end] that the NYSE calendar says were open."""
    holidays = market_holidays(start, end)
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur not in holidays:
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CleaningReport:
    """Everything the caller needs to judge the file, with nothing repaired."""

    rows_in_file: int
    rows_in_window: int
    trading_days_expected: int
    trading_days_observed: int
    blank_weekdays: int
    blank_weekdays_matching_a_holiday: int
    blank_on_an_expected_trading_day: tuple[str, ...]
    weekday_absent_from_file: tuple[str, ...]
    unexpected_value_on_a_holiday: tuple[str, ...]
    duplicate_dates: tuple[str, ...]
    nonpositive_prices: tuple[str, ...]
    out_of_order: bool
    coverage_requested: tuple[str, str]
    coverage_available: tuple[str, str]
    covers_request: bool
    dropped_nontrading_observations: tuple[str, ...] = ()

    def defects(self) -> list[str]:
        msgs = []
        if self.duplicate_dates:
            msgs.append(f"{len(self.duplicate_dates)} duplicate date(s)")
        if self.nonpositive_prices:
            msgs.append(f"{len(self.nonpositive_prices)} non-positive price(s)")
        if self.out_of_order:
            msgs.append("dates are not in ascending order in the raw file")
        if self.blank_on_an_expected_trading_day:
            msgs.append(f"{len(self.blank_on_an_expected_trading_day)} blank(s) on a "
                        "day the calendar says the market was open")
        if self.weekday_absent_from_file:
            msgs.append(f"{len(self.weekday_absent_from_file)} weekday row(s) missing "
                        "from the file entirely")
        if self.unexpected_value_on_a_holiday:
            how = ("dropped at the analysis layer, the next return re-formed from the "
                   "adjacent valid closes"
                   if self.dropped_nontrading_observations else "KEPT in the series")
            msgs.append(f"{len(self.unexpected_value_on_a_holiday)} price(s) printed "
                        "on a day the calendar says the exchange was closed ("
                        + ", ".join(self.unexpected_value_on_a_holiday[:5])
                        + f") -- {how}")
        if not self.covers_request:
            msgs.append("the source does not cover the requested window")
        return msgs

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()} | {"defects": self.defects()}


def clean_price_series(raw: pd.DataFrame, start: dt.date, end: dt.date,
                       calendar: str = "nyse",
                       drop_nontrading_observations: bool = True
                       ) -> tuple[pd.DataFrame, CleaningReport]:
    """Trading days with a positive price, plus a report of everything checked.

    Nothing is filled, interpolated or forward-filled.  A holiday does not become
    a zero return, and a blank is never turned into a price.
    """
    rows_in_file = int(len(raw))
    out_of_order = bool((raw["date"].diff().dropna() <= pd.Timedelta(0)).any())
    window = raw[(raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))]
    window = window.sort_values("date", ignore_index=True)

    dup = window["date"][window["date"].duplicated()].dt.date.astype(str)
    have = window.dropna(subset=["value"]).reset_index(drop=True)
    nonpos = have.loc[have["value"] <= 0, "date"].dt.date.astype(str)

    present = {d.date() for d in window["date"]}
    observed = {d.date() for d in have["date"]}
    blanks = present - observed

    if calendar not in CALENDARS:
        raise ValueError(f"unknown calendar mode {calendar!r}")
    # the expected set comes from an INDEPENDENT rule set, never from the dates that
    # happen to be observed -- otherwise the check can never fail
    expected = expected_trading_days_for(calendar, start, end)
    holidays = CALENDARS[calendar](start, end)
    expected_set = set(expected)

    blank_on_open = sorted(str(d) for d in (blanks & expected_set))
    absent_weekday = sorted(str(d) for d in expected_set - present)
    value_on_holiday = sorted(str(d) for d in (observed & set(holidays)))

    avail = (str(have["date"].min().date()), str(have["date"].max().date())) if len(have) \
        else ("", "")
    covers = bool(len(have)) and have["date"].min() <= pd.Timestamp(start) + pd.Timedelta(days=7) \
        and have["date"].max() >= pd.Timestamp(end) - pd.Timedelta(days=7)

    report = CleaningReport(
        rows_in_file=rows_in_file,
        rows_in_window=int(len(window)),
        trading_days_expected=len(expected),
        trading_days_observed=int(len(have)),
        blank_weekdays=int(len(blanks)),
        blank_weekdays_matching_a_holiday=int(len(blanks & set(holidays))),
        blank_on_an_expected_trading_day=tuple(blank_on_open),
        weekday_absent_from_file=tuple(absent_weekday),
        unexpected_value_on_a_holiday=tuple(value_on_holiday),
        duplicate_dates=tuple(dup),
        nonpositive_prices=tuple(nonpos),
        out_of_order=out_of_order,
        coverage_requested=(str(start), str(end)),
        coverage_available=avail,
        covers_request=covers,
        dropped_nontrading_observations=tuple(value_on_holiday)
        if drop_nontrading_observations else (),
    )
    clean = have.rename(columns={"value": "price"})[["date", "price"]]
    if drop_nontrading_observations and value_on_holiday:
        # The RAW SNAPSHOT KEEPS THE ROW. Here, at the analysis layer, a price
        # printed on a day the calendar says the exchange was closed is removed, so
        # the next return is formed from the adjacent VALID closes instead of
        # treating the artefact as an ordinary trading day.
        clean = clean[~clean["date"].dt.date.astype(str).isin(value_on_holiday)]
    return clean.reset_index(drop=True), report


def simple_returns(prices: pd.DataFrame, calendar: str = "nyse") -> pd.DataFrame:
    """r_t = P_t / P_{t-1} - 1, as a DECIMAL, between consecutive observed days.

    ``gap_trading_days`` counts how many trading days the calendar says the step
    should have spanned.  It is 1 for an ordinary day.  Anything larger means the
    step jumped a day the market was open but the file did not report, and
    ``spans_missing_day`` marks it so that it is never scored as a one-day return.
    """
    out = prices.sort_values("date", ignore_index=True).copy()
    out["prev_date"] = out["date"].shift(1)
    out["ret"] = out["price"] / out["price"].shift(1) - 1.0
    out = out.dropna(subset=["ret"]).reset_index(drop=True)

    gaps = []
    for prev, cur in zip(out["prev_date"], out["date"]):
        gaps.append(len(expected_trading_days_for(
            calendar, prev.date() + dt.timedelta(days=1), cur.date())))
    out["gap_trading_days"] = np.asarray(gaps, dtype=int)
    out["spans_missing_day"] = out["gap_trading_days"] > 1
    return out[["date", "prev_date", "price", "ret", "gap_trading_days",
                "spans_missing_day"]]
