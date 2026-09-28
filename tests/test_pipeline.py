import numpy as np  # noqa: F401

from purchase_value import config as C
from purchase_value.preprocessing import build_preprocessor, clean
from purchase_value.train import fit_eval_predict, prepare


def test_clean_drops_columns_and_adds_date_parts(train_df):
    out = clean(train_df)
    assert "date" not in out and {"month", "year"} <= set(out.columns)
    assert not set(C.DROP_COLUMNS) & set(out.columns)
    assert out["os"].isna().any()  # "(not set)" became NaN


def test_preprocessor_outputs_numeric_matrix_and_handles_unseen_categories(train_df, test_df):
    X = clean(train_df).drop(columns=C.TARGET)
    pre = build_preprocessor(X)
    Xt = pre.fit_transform(X)
    test_df.loc[:, "browser"] = "SomethingNew"
    Xn = pre.transform(clean(test_df))
    assert list(Xn.columns) == list(Xt.columns)
    assert not Xn.isna().any().any()
    assert (Xt.dtypes == float).all()


def test_end_to_end_predictions_are_valid(train_df, test_df):
    X, y, test = prepare(train_df, test_df)
    preset = dict(C.XGB_PRESETS["regularised"], n_estimators=20)
    C.XGB_PRESETS["tiny"] = preset
    _, _, preds = fit_eval_predict(X, y, test, "tiny", rfe_features=10)
    assert preds.shape == (len(test_df),)
    assert (preds >= 0).all()
