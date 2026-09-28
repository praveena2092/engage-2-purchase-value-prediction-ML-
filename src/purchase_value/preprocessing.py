"""Cleaning + feature engineering + the sklearn preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler

from . import config as C


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder strings -> NaN, drop useless columns, derive month/year from `date`."""
    df = df.replace(C.NULL_TOKENS, np.nan)
    df = df.drop(columns=[c for c in C.DROP_COLUMNS if c in df.columns])
    if "date" in df.columns:
        date = pd.to_datetime(df["date"], format="%Y%m%d")
        df["month"] = date.dt.month
        df["year"] = date.dt.year
        df = df.drop(columns="date")
    return df


def _to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Imputing mixed columns leaves some as object dtype (the notebook cast one by hand)."""
    return df.astype(float)


def _present(cols: list[str], frame: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in frame.columns]


def build_preprocessor(sample: pd.DataFrame) -> Pipeline:
    """impute -> encode -> scale, built from the columns actually present in `sample`."""
    impute = ColumnTransformer(
        [
            ("mode", SimpleImputer(strategy="most_frequent", keep_empty_features=True), _present(C.IMPUTE_MOST_FREQUENT, sample)),
            ("const", SimpleImputer(strategy="constant", fill_value="constant", keep_empty_features=True), _present(C.IMPUTE_CONSTANT_CATEGORY, sample)),
            ("zero", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True), _present(C.IMPUTE_ZERO, sample)),
            ("true", SimpleImputer(strategy="constant", fill_value=True, keep_empty_features=True), _present(C.IMPUTE_TRUE, sample)),
            ("false", SimpleImputer(strategy="constant", fill_value=False, keep_empty_features=True), _present(C.IMPUTE_FALSE, sample)),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    encode = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="infrequent_if_exist"), _present(C.OHE, sample)),
            ("ohe_top5", OneHotEncoder(sparse_output=False, drop="first", max_categories=6, handle_unknown="infrequent_if_exist"), _present(C.OHE_TOP5, sample)),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), _present(C.ORDINAL, sample)),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scale = ColumnTransformer(
        [("robust", RobustScaler(), _present(C.ROBUST_SCALE, sample))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("impute", impute),
            ("encode", encode),
            ("scale", scale),
            ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
        ]
    ).set_output(transform="pandas")
