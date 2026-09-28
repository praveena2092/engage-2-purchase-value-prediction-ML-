"""Train an XGBoost regressor, report hold-out R2, and write a Kaggle submission CSV.

    python -m purchase_value.train --preset notebook --rfe 25
    python -m purchase_value.train --preset regularised
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from . import config as C
from .preprocessing import build_preprocessor, clean


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / C.TRAIN_FILE)
    test = pd.read_csv(data_dir / C.TEST_FILE)
    return train, test


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    train = clean(train).drop_duplicates()
    test = clean(test)
    y = train[C.TARGET]
    X = train.drop(columns=C.TARGET)
    return X, y, test


def fit_eval_predict(X, y, test, preset: str, rfe_features: int | None, val_size: float = 0.2):
    params = C.XGB_PRESETS[preset]

    def fit(X_fit, y_fit):
        pre = build_preprocessor(X_fit)
        Xt = pre.fit_transform(X_fit)
        selector = None
        if rfe_features:
            selector = RFE(XGBRegressor(**params), n_features_to_select=rfe_features)
            Xt = selector.fit_transform(Xt, y_fit)
        model = XGBRegressor(**params).fit(Xt, y_fit)
        return pre, selector, model

    def predict(parts, X_new):
        pre, selector, model = parts
        Xt = pre.transform(X_new)
        if selector is not None:
            Xt = selector.transform(Xt)
        return np.clip(model.predict(Xt), 0, None)  # purchaseValue is never negative

    # 1) honest hold-out estimate
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_size, random_state=C.RANDOM_STATE)
    parts = fit(X_tr, y_tr)
    val_r2 = r2_score(y_val, predict(parts, X_val))
    train_r2 = r2_score(y_tr, predict(parts, X_tr))

    # 2) refit on everything, predict test
    parts = fit(X, y)
    return train_r2, val_r2, predict(parts, test)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--preset", choices=list(C.XGB_PRESETS), default="regularised")
    ap.add_argument("--rfe", type=int, default=None, help="keep the top-N features chosen by RFE")
    ap.add_argument("--out", type=Path, default=C.SUBMISSION_DIR / "submission.csv")
    args = ap.parse_args()

    train, test = load(args.data_dir)
    X, y, test = prepare(train, test)
    print(f"train {X.shape}, test {test.shape}")

    train_r2, val_r2, preds = fit_eval_predict(X, y, test, args.preset, args.rfe)
    print(f"R2 train (fit part): {train_r2:.4f} | R2 hold-out: {val_r2:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": range(len(preds)), C.TARGET: preds}).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
