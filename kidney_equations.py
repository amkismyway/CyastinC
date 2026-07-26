"""Kidney-function equations and NHANES calibration helpers."""

from __future__ import annotations
import numpy as np
import pandas as pd


def calibrate_creatinine(reported: pd.Series, cycle: pd.Series) -> pd.Series:
    """Standardize NHANES creatinine; correct 1999-2000 values only."""
    reported = pd.to_numeric(reported, errors="coerce")
    return pd.Series(
        np.where(cycle.eq("1999-2000"), 0.147 + 1.013 * reported, reported),
        index=reported.index,
        name="creatinine_calibrated",
    )


def calibrate_cystatin_c(reported: pd.Series) -> pd.Series:
    """Convert legacy NHANES cystatin C to the IFCC-traceable scale."""
    reported = pd.to_numeric(reported, errors="coerce")
    calibrated = 1.12 * (reported - 0.12)
    return calibrated.where(calibrated > 0)


def egfr_creatinine_2021(creatinine: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    """Calculate race-free 2021 CKD-EPI creatinine eGFR for adults."""
    creatinine = pd.to_numeric(creatinine, errors="coerce")
    age = pd.to_numeric(age, errors="coerce")
    female = pd.Series(female, index=creatinine.index).fillna(0).astype(bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = creatinine / kappa
    result = (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** -1.200
        * 0.9938 ** age
        * np.where(female, 1.012, 1.0)
    )
    invalid = creatinine.isna() | age.isna() | (creatinine <= 0) | (age < 18)
    return pd.Series(np.where(invalid, np.nan, result), index=creatinine.index)


def egfr_cystatin_2012(cystatin_c: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    """Calculate CKD-EPI cystatin C eGFR for adults."""
    cystatin_c = pd.to_numeric(cystatin_c, errors="coerce")
    age = pd.to_numeric(age, errors="coerce")
    female = pd.Series(female, index=cystatin_c.index).fillna(0).astype(bool)
    ratio = cystatin_c / 0.8
    result = (
        133
        * np.minimum(ratio, 1) ** -0.499
        * np.maximum(ratio, 1) ** -1.328
        * 0.996 ** age
        * np.where(female, 0.932, 1.0)
    )
    invalid = cystatin_c.isna() | age.isna() | (cystatin_c <= 0) | (age < 18)
    return pd.Series(np.where(invalid, np.nan, result), index=cystatin_c.index)
