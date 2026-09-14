# Data source inventory

Every object the supervisor has named, with what would actually have to arrive for it
to enter a study. Replaces the earlier shortlist; companion to
[`MARKET_STAGE1_PROTOCOL.md`](MARKET_STAGE1_PROTOCOL.md),
[`MARKET_DATA_DICTIONARY.md`](MARKET_DATA_DICTIONARY.md) and
[`CROSS_MARKET_SCAN_SPEC.md`](CROSS_MARKET_SCAN_SPEC.md).

**Status key.** *held* — archived locally with a SHA-256. *obtainable* — one command
with the existing downloader. *needs access* — a specific request, listed rather than
worked around. *ambiguous* — the meeting record names a category, not an instrument;
the gap is written down rather than resolved by guessing.

## Rules that apply to every row

1. A monthly series is never interpolated to daily.
2. Returns, rate levels and implied volatilities are never ranked or pooled together.
   A return is a change; a yield is a level in percent per annum; an implied
   volatility is an annualised volatility in percent or in basis points.
3. The quoting convention is recorded before a series is used.
4. A price index and a total-return index never appear in the same comparison.
5. No series is stitched from more than one vendor.

---

## 0. What is actually in the repository

The numbered sections below describe **categories**, including several we do not hold.
This table is the registry itself (`market_assets.ASSETS`), so it cannot drift from the
code: sixteen objects, three measurement categories, eight of them in the model
contrast.

| object | category | what the number is | in the model contrast | source |
|---|---|---|---|---|
| Momentum factor, daily (`ff_momentum`) | `return` | a return on a strategy, factor or vehicle | yes | Kenneth French |
| SPDR Gold Shares (`gld`) | `return` | a return on a strategy, factor or vehicle | yes | SPDR official XLSX |
| S&P 500 (`sp500`) | `price_change` | a price or rate change, **not** an investment return | yes | FRED `SP500` |
| Nasdaq-100 (`nasdaq100`) | `price_change` | a price or rate change, **not** an investment return | yes | FRED `NASDAQ100` |
| Nikkei 225 (`nikkei225`) | `price_change` | a price or rate change, **not** an investment return | yes | FRED `NIKKEI225` |
| EURO STOXX 50 (`sx5e`) | `price_change` | a price or rate change, **not** an investment return | **no** | STOXX archive, ends 2016-10-04 |
| Brent crude spot (`brent`) | `price_change` | a price or rate change, **not** an investment return | yes | FRED `DCOILBRENTEU` |
| EUR/USD spot (`eurusd`) | `price_change` | a price or rate change, **not** an investment return | yes | FRED `DEXUSEU` |
| USD/JPY spot (`usdjpy`) | `price_change` | a price or rate change, **not** an investment return | yes | FRED `DEXJPUS` |
| VIX (`vix`) | `market_state` | a level describing market state; levels and changes only | no | Cboe official |
| OVX, crude implied vol (`ovx`) | `market_state` | a level describing market state; levels and changes only | no | Cboe official |
| GVZ, gold implied vol (`gvz`) | `market_state` | a level describing market state; levels and changes only | no | Cboe official |
| US 2y yield (`dgs2`) | `market_state` | a level describing market state; levels and changes only | no | FRED `DGS2` |
| US 10y yield (`dgs10`) | `market_state` | a level describing market state; levels and changes only | no | FRED `DGS10` |
| US 30y yield (`dgs30`) | `market_state` | a level describing market state; levels and changes only | no | FRED `DGS30` |
| WTI crude spot (`wti`) | `market_state` | a level describing market state; levels and changes only | no | FRED `DCOILWTICO` |

WTI is a `market_state` series rather than a `price_change` one for a specific reason:
FRED's `DCOILWTICO` prints **-36.98 on 2020-04-20**. A simple return across a sign
change is not a meaningful quantity, so the level is kept as printed — not deleted, not
winsorised, not log-transformed — and no return series is formed from it.

**Hold-out, stated once and the same way everywhere.** 2024-2025 takes **no part in
any statistic, figure, parameter estimate, or choice of scheme**. Separately, and as a
matter of implementation rather than policy: snapshot files are read **whole**, so rows
dated 2024-2025 are present in memory, and the study window is applied after loading.
The bytes pass through; no value in them reaches a number that is reported or acted on.


## 1. Equity price indices

| | S&P 500 | Nasdaq-100 | Nikkei 225 | Euro Stoxx 50 |
|---|---|---|---|---|
| observation variable | daily closing index level | daily closing index level | daily closing index level | daily closing index level |
| units | index points | index points | index points (JPY) | index points (EUR) |
| price or total return | **price**, dividends excluded | **price**, dividends excluded | **price**, dividends excluded | **price**, settled by reading the STOXX file: its title is *Price Indices - EURO Currency* and it carries no TR column. SX5T is the total-return twin and is not in that file |
| frequency | daily, business days, blank on US market holidays | same | daily, blank on Japanese market holidays | daily, blank on Euronext/Xetra holidays |
| available history | FRED keeps a **rolling 10 years** only | 1986-01-02 onward | 1949-05-16 onward | public STOXX archive 1986-12-31..**2016-10-04**; nothing in 2017-2025 |
| source | FRED `SP500` | FRED `NASDAQ100` | FRED `NIKKEI225` | STOXX `hbrbcpe.txt` is public and its own header names column 2 `SX5E`, but the archive **ends 2016-10-04**. Not on FRED either (`STOXX50E`, `SX5E`, `EUROSTOXX50` → 404), though a 404 on a guessed id was never evidence of unavailability |
| permission | free to download; S&P DJI data **not redistributable** | free; Nasdaq index data | free; Nikkei index data | vendor or exchange licence |
| obtainable now | **held** | **held** | **held** | **needs access** |

**Defect to carry forward.** FRED's `NASDAQ100` prints a price on **2019-04-19**, Good
Friday, when the US equity market was closed; `SP500` correctly leaves it blank. The
row is kept and flagged, never deleted. **Calendar limitation, resolved.** This entry
formerly said no encoded JPX calendar existed, so the Nikkei's non-trading days came
from the source file's own blank rows with nothing independent to check them against.
A JPX rule set is now encoded in `market_data.py` and agrees with the source blanks on
**114 of 114** days over 2017-2023, so the Nikkei is cross-validated like the US series.

## 2. Equity implied volatility

| | VIX |
|---|---|
| observation variable | daily close of the VIX index |
| units | **percent, annualised implied volatility**. A level of 20 means about 20% annualised, **not** a 20% return. A daily change is in **VIX points**, a change in an annualised volatility, and is not a return on anything |
| frequency | daily, business days |
| available history | 1990-01-02 onward |
| source | FRED `VIXCLS`; Cboe's own history page is the alternative |
| permission | free to download; Cboe index data |
| obtainable now | **held** |

Used as a description only. It looks **forward** about 30 calendar days while `RV_21`
looks **back** 21 trading days, so the two are not required to agree point by point.
It is never substituted for the simulator's latent variance, and its percentage
changes are never treated as equity returns.

## 3. Swaption volatility

| | USD 1M × 10Y ATM swaption implied volatility |
|---|---|
| observation variable | daily close of the at-the-money implied volatility for a 1-month option on a 10-year swap |
| units | **must be stated on delivery**: normal (basis points per annum) or lognormal/Black (percent). They differ by roughly the forward rate and are not interchangeable |
| frequency | daily |
| available history | whatever the licence allows; 2017-2025 is the minimum useful span |
| source | Bloomberg (`USSN0110` normal, `USSV0110` Black), Refinitiv/ICAP, CME, or a bank research feed |
| permission | licensed vendor data; not downloadable by us |
| obtainable now | **needs access** |

**The specific ask.** One CSV with: date; ATM implied volatility; an explicit statement
of **normal versus Black**; the forward swap rate on the same dates if available; the
day count and holiday calendar used. **A Treasury yield series is not a substitute for
this and will not be used as one** — a yield level is not a volatility.

## 4. FX

| | FX total return |
|---|---|
| observation variable | **ambiguous in the meeting record.** "FX total return" names a category, not an instrument |
| what has to be decided first | which pair or basket; spot-only or spot plus carry; funded in which currency; if a basket, its weights and rebalancing rule |
| units | a return series in decimals once the above is fixed |
| frequency | daily |
| source | depends entirely on the answer above |
| permission | unknown until the instrument is named |
| obtainable now | **ambiguous** — not resolved by guessing an instrument |

A carry-funded FX total return and a spot return are different series with different
volatility shapes; picking one on our own would silently choose the answer.

## 5. Commodities

| | Commodity QIS |
|---|---|
| observation variable | **ambiguous in the meeting record.** "Commodity QIS" names a product family, not a series |
| what has to be decided first | which strategy (carry, momentum, curve, inventory); which provider's index; gross or net of fees and financing; the inception date and whether pre-inception history is backtested |
| units | a return series in decimals |
| frequency | daily if it exists at daily |
| source | the issuing bank's index feed, or internal |
| permission | internal or vendor |
| obtainable now | **ambiguous / needs access** |

If any part of the history is backtested rather than live, that must be stated: a
backtested segment and a live segment are not the same kind of data for this project.

## 6. Factor returns

| | Kenneth French daily momentum | AQR factor sets |
|---|---|---|
| observation variable | daily MOM factor return | factor returns (BAB, QMJ, TSMOM, Century of Factor Premia) |
| units | **percent** in the published file; converted to decimal once on load | percent in the published files |
| price or return | a **return** on a zero-investment long-short portfolio; not a price, not net of cost | same |
| frequency | daily | **mostly monthly**; daily files exist for some sets — check per file |
| available history | 1926-11-03 onward | varies by file |
| source | Kenneth French data library | AQR data library |
| permission | free for research use | free for research use subject to their terms; confirm whether firm use qualifies |
| obtainable now | **held** | **obtainable**, per file |

The French definition, recorded so it is not restated loosely: MOM is the average
return of the two high-prior-return value-weight portfolios minus the average of the
two low-prior-return portfolios, prior return measured from day −250 to −21.

## 7. Interest rates

| | US Treasury constant maturity and overnight rates |
|---|---|
| observation variable | daily yield or rate |
| units | **percent per annum**, a level and not a return |
| frequency | daily, business days |
| available history | `DGS10` from 1962; `SOFR` only from April 2018 |
| source | FRED `DGS1MO`…`DGS30`, `SOFR`, `EFFR` |
| permission | US government data, freely redistributable |
| obtainable now | **obtainable**, not yet archived |

Two candidate uses, both out of scope so far: a risk-free leg if the project ever
moves from price returns to excess returns, and a second asset class for a volatility
shape check. If a "rate return" is ever needed, decide and record whether it means a
yield change in basis points or a bond total return.

## 8. Company strategy data

| | Internal QIS / strategy daily returns |
|---|---|
| observation variable | daily net return per strategy |
| units | decimals |
| frequency | daily |
| available history | whatever the firm retains |
| source | internal |
| permission | internal, likely a data-governance sign-off |
| obtainable now | **needs access** |

**The specific asks.** Is the series net of costs and financing? Does the start date
mark live trading or backtest inception? Is there a recorded decision to stop or
downsize a strategy, and its date? Are decommissioned strategies retained in the file,
or is it current-only? The last question decides what can be concluded at all: a
current-only file is the survivorship problem this project is named after.

**What a stop record is, and is not.** It is an operational event with a date: somebody
decided to cut or shrink a strategy. Treated as a label it is a **proxy**, and a noisy
one in both directions — capacity limits, risk budgets and mandate changes end sound
strategies, and unsound ones survive inattention. So it is not a true Sharpe ratio, not
a verdict on validity, and not a true date of failure. What makes it valuable is that it
is **dated and real**: a detection rule can be scored against when a decision was
actually taken, which is more than any index in this repository offers.

---

## What is in the first simulation batch, and why

Fixed before any result was seen: **S&P 500**, **Nasdaq-100**, **Nikkei 225** and the
**Kenneth French daily momentum factor**. The momentum factor is there to add a
*strategy-return* type alongside three price indices — not because it fits. Euro Stoxx
50 was in the plan and is **not** in the batch. The reason is coverage, not access:
the public STOXX file is reachable and its SX5E column is confirmed by the file's own
header, but it ends 2016-10-04 and cannot reach the 2017-2025 window. The gap is
recorded here and the slot was **not** filled with a better-fitting substitute.

## What would change a conclusion, ranked

1. **Internal strategy returns with a live start date and a stop record.** Everything
   so far constrains the noise side only. Note what such a file would and would not
   supply: a stop or downsize decision is an **operational event**, and at best a proxy
   for a strategy having failed. It is not the true Sharpe ratio, not a verdict on
   validity, and not a true failure date — a strategy can be cut for capacity, risk
   budget, staff turnover or a mandate change while still being sound, and an unsound
   one can run for years untouched. The file would give a **dated decision**, which is
   what a detection study can be scored against; the label itself stays unobserved.
2. **Euro Stoxx 50 or another non-US, non-Japanese index.** The current batch has one
   non-US object; a second would separate "different market" from "different calendar".
3. **A daily AQR factor set.** More strategy-type returns, free, and a second
   strategy-type object to read the momentum factor against. The earlier reason given
   here — that momentum shows unusually slow volatility decay — is **withdrawn**: its
   lag-63 absolute-return autocorrelation is +0.164 in 2017-2021 and -0.013 in
   2022-2023, and both sit inside the fixed model's simulated range.
4. **Swaption volatility and rates.** A cross-asset check, but they answer no question
   the project currently has.

None of these is the next round's work. That is specified in
`BENCHMARK_PROTOCOL_NEXT_ROUND.md` and needs no new data: it benchmarks the
incumbent noise model against iid Student-t and GARCH(1,1)-t on the S&P history
we already hold.
