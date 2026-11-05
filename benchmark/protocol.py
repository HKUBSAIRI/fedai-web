"""Estimation protocol shared by the benchmark scripts.
Default: expanding window - every SEP re-estimates the parameters on all earlier SEPs.
FEDAI_TRAIN_END=YYYY-MM-DD (the group's unified protocol uses 2015-12-31): parameters - regression coefficients, the
herding share, the choice of method in B13 - are estimated only on SEPs up to that date and stay fixed afterwards. The
history still updates meeting by meeting: own previous dots, personal averages of past submissions, macro data."""
import os
import pandas as pd
TRAIN_END = pd.Timestamp(os.environ['FEDAI_TRAIN_END']) if os.environ.get('FEDAI_TRAIN_END') else None
def fitset(df, col='meeting_date'):
    """Rows that may be used to estimate parameters."""
    return df if TRAIN_END is None else df[df[col] <= TRAIN_END]
def may_learn(t):
    """Whether the outcome of SEP t may still change estimated parameters (e.g. B13's record of past errors)."""
    return TRAIN_END is None or pd.Timestamp(t) <= TRAIN_END
