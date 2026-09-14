# Data request

Ordered by how much each would change a conclusion. Everything public that could be
fetched has been fetched; what follows is what cannot be.

| # | item | status | what it unblocks |
|---|---|---|---|
| 1 | USD 1M x 10Y ATM swaption volatility | no public substitute | the only rates-vol object; nothing we hold can stand in |
| 2 | FX carry total return, and commodity QIS | instruments not named in the meeting record | two of the four asset categories the scan is supposed to span |
| 3 | broad commodity benchmark / FX total return | as above | gives the gold and crude spot series something investable to be read against |
| 4 | EURO STOXX 50 daily, 2017-2025 | **source found, coverage insufficient** | a second non-US index, separating "different market" from "different calendar" |
| 5 | internal strategy returns | not public | dated stop/downsize decisions — a real event to score detection against, **not** a true validity label |
| 6 | your designated file, or any additional daily factors | **not yet obtained** | a second strategy-type return, so momentum is not the sole example of one |

Items 2 and 3 are blocked on a choice of instrument, not on access. Item 4 is blocked
on coverage, not on access. We are not filling any of them with a substitute.

## 1. Swaption volatility — USD 1M × 10Y ATM

The single item with no public substitute. **A Treasury yield is not a swaption
volatility and will not be used as one.**

Please supply one CSV with, per row: date; the ATM implied volatility; and, stated
once in the header or a covering note:

- **normal (basis points per annum) or lognormal/Black (percent)** — they differ by
  roughly the forward rate and are not interchangeable;
- the **currency** (USD assumed, please confirm);
- the **timestamp convention** — London or New York close, and which time;
- the **underlying rate** the swap is on (SOFR OIS, or LIBOR before the transition);
- any **historical convention change**, in particular the LIBOR-to-SOFR transition
  date and whether the series was restated across it or spliced.

Ticker candidates if it comes from Bloomberg: `USSN0110` (normal), `USSV0110` (Black).

## 2. FX and commodity QIS — you choose the instruments

The meeting record names categories, not instruments, so these are listed as
**ambiguous** rather than resolved by us picking something. **An FX spot change is not
an FX carry total return, and GLD or crude spot is not a commodity QIS** — we hold the
spot series and have labelled them as spot, not as investable returns.

For each series you select, please give: the exact code; whether the file is a daily
**level** or a daily **return**; gross or net of **costs**; whether **financing** is
included; the **roll** rule for anything futures-based; whether a **volatility target**
is applied and at what level; and the **live launch date** with the boundary against
any backtested history before it.

## 3. Broad commodity benchmark and FX total return

Same fields as item 2. A broad commodity ER or TR index would let the gold and crude
spot series we already hold be read against something investable.

## 4. EURO STOXX 50, daily, 2017-2025

Status: **source found, column confirmed, coverage insufficient.** Three separate
facts, and only the third is a problem:

1. The public STOXX file `hbrbcpe.txt` is reachable without a licence.
2. Its own header row names the columns `SX5P | SX5E | SXXP | SXXE | ...`, so the
   EURO STOXX 50 column is **confirmed by the file itself**, not inferred from a
   guessed ticker. Its title, *Price Indices - EURO Currency*, also settles that it is
   a EUR **price** index with no total-return column in the file.
3. That archive **ends 2016-10-04**. It cannot reach the 2017-2025 study window, so
   there is still **no in-window daily history** for this object.

What would close it: the companion current-period STOXX file, or a licensed feed
covering 2017-2025. Please state the index variant (SX5E price or SX5T total return),
the currency, and the redistribution terms.

(The previous round's claim that Euro Stoxx was unavailable came from three guessed
FRED ids returning 404. That inference was wrong, and a 404 on a guessed id is not
evidence of anything; the coverage end date above is the real reason.)

## 5. Internal strategy returns

Please confirm: net of costs and financing; live-trading start versus backtest
inception; whether stop or downsize decisions and their dates are recorded; and whether
decommissioned strategies are retained in the file or it is current-only.

**On what a stop record actually is.** Earlier versions of this page called it "the only
source that could carry the effective/ineffective label". That overstates it. A stop or
downsize is an **operational event** — dated, real, and taken for reasons that include
capacity, risk budget, mandate and staffing as well as performance. As a label for
validity it is a **proxy** and noisy in both directions: sound strategies get cut, and
unsound ones survive inattention. It is not a true Sharpe ratio and not a true date of
failure.

What makes it worth asking for anyway is the **date**. A detection rule can be scored
against when a decision was actually taken, which no index in this repository offers.
The true Sharpe ratio stays unobserved in real data; that is why the simulator with a
known label exists alongside it, and neither replaces the other.

## 6. The daily factor gap: your designated file, or additional daily factors

**Status: not yet obtained.** This is a gap in what we hold, not a claim about what you
have — we do not know what is on your disk, and nothing here should be read as saying
your holdings are limited to files we happen to have downloaded ourselves.

What is already closed: **Kenneth French daily momentum is obtained, cleaned, and in
the daily model contrast** as `ff_momentum`, 1,259 returns over 2017-2021 on the NYSE
calendar. It is not pending.

What is still open: it is the **only** strategy-type return in the batch. Of the eight
contrast objects, six are price changes and one is an ETF price return, so every
statement about "how a strategy behaves" currently rests on one series. A second daily
factor would separate "a property of momentum" from "a property of factor returns".

Per file, please say: the construction (long-short or long-only, and the universe);
**gross or net of costs**; daily or monthly — a monthly file enters as a monthly
supplement and is never interpolated to daily; the sample start and whether any part of
it is backtested rather than live; and whether the file may leave this machine.

---

## What we already hold, so you do not need to send it

S&P 500, Nasdaq-100, Nikkei 225 (FRED); Kenneth French daily momentum; SPDR Gold
Shares official history; Brent and WTI spot, EUR/USD and USD/JPY spot, US 2y/10y/30y
Treasury yields (FRED); VIX, OVX, GVZ (Cboe official); AQR time-series momentum and
commodities-long-run files — both confirmed **monthly** by reading the files, so they
enter as a monthly supplement only and are never interpolated to daily.

That AQR line describes **what we downloaded**, and says nothing about what you hold.
It is listed so you do not send it twice, not as an inventory of the factor data in
existence.

**Hold-out.** 2024-2025 takes no part in any statistic, figure, parameter estimate or
choice of scheme. As an implementation detail, snapshot files are read whole, so those
rows sit in memory and the window is applied after loading; no value in them reaches a
reported number.

## Two access notes for whoever runs this next

FRED **refuses** a spoofed desktop-browser User-Agent — all seven rate, energy and FX
requests failed with it and succeeded immediately with the default agent. Cboe, SPDR
and STOXX need the opposite: a browser agent and a `Referer`. Both are recorded in
`data/raw/acquisition.provenance.json`.
