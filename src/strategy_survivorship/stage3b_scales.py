"""Which statistics live on a log-odds scale.

Kept in its own module so both the horizon diagnostic and the failure-time
diagnostic can import it without a circular import.  The trailing Sharpe and the
fixed-sigma rolling control produce annualised Sharpe ratios, not log-odds:
mapping them through ``expit`` yields a number with no interpretation, so every
probability-scale field must be left blank for them.
"""

LOG_ODDS_METHODS = ("binary_gaussian", "binary_student_t", "ewma_gaussian",
                    "ewma_student_t", "ewma_trunc_gaussian", "ewma_trunc_student_t")
NOT_LOG_ODDS = ("trailing_sharpe_252", "known_vol_rolling_252")
