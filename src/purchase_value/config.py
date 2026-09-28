"""Column groups and settings, ported from notebooks/01_eda_preprocessing_modelling.ipynb."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"
SUBMISSION_DIR = ROOT / "submissions"

TARGET = "purchaseValue"
RANDOM_STATE = 42

# Placeholder strings that really mean "missing".
NULL_TOKENS = [
    "not available in demo dataset",
    "(not set)",
    "(not provided)",
    "Not Socially Engaged",
    "(none)",
]

# Columns that are empty / constant once placeholders become NaN, or that are IDs.
ALL_NAN_COLUMNS = [
    "device.screenResolution", "device.mobileDeviceBranding", "device.mobileInputSelector",
    "device.mobileDeviceMarketingName", "device.operatingSystemVersion", "device.flashVersion",
    "browserMajor", "device.browserSize", "socialEngagementType", "device.mobileDeviceModel",
    "device.language", "device.browserVersion", "device.screenColors", "geoNetwork.networkLocation",
]
HIGH_CARDINALITY_OR_CONSTANT = ["sessionId", "userId", "sessionStart", "locationZone", "screenSize"]
DROP_COLUMNS = ALL_NAN_COLUMNS + HIGH_CARDINALITY_OR_CONSTANT

# ── Imputation groups ──────────────────────────────────────────────────────
IMPUTE_MOST_FREQUENT = ["pageViews", "browser", "geoCluster", "trafficSource.adwordsClickInfo.page"]
IMPUTE_CONSTANT_CATEGORY = [
    "trafficSource.referralPath", "geoNetwork.city", "geoNetwork.subContinent", "trafficSource.medium",
    "locationCountry", "geoNetwork.metro", "geoNetwork.continent", "geoNetwork.region", "os",
    "trafficSource.adContent", "trafficSource.keyword", "trafficSource.adwordsClickInfo.slot",
    "trafficSource.campaign", "trafficSource.adwordsClickInfo.adNetworkType",
]
IMPUTE_ZERO = ["totals.bounces", "new_visits"]
IMPUTE_TRUE = ["trafficSource.adwordsClickInfo.isVideoAd"]
IMPUTE_FALSE = ["trafficSource.isTrueDirect"]

# ── Encoding groups ────────────────────────────────────────────────────────
# High-cardinality: keep the top 5 categories, group the rest as "infrequent".
OHE_TOP5 = [
    "browser", "trafficSource.adContent", "trafficSource.keyword", "trafficSource.campaign",
    "geoNetwork.region", "trafficSource", "os", "geoNetwork.subContinent", "locationCountry",
    "geoNetwork.city", "geoNetwork.metro", "trafficSource.referralPath", "month",
]
OHE = [
    "geoCluster", "trafficSource.adwordsClickInfo.slot", "geoNetwork.networkDomain",
    "trafficSource.medium", "deviceType", "userChannel", "geoNetwork.continent",
    "device.isMobile", "year",
]
ORDINAL = [
    "trafficSource.isTrueDirect", "trafficSource.adwordsClickInfo.isVideoAd",
    "trafficSource.adwordsClickInfo.adNetworkType",
]
ROBUST_SCALE = ["sessionNumber", "pageViews"]

# ── Model presets ──────────────────────────────────────────────────────────
XGB_PRESETS = {
    # Settings behind the submission recorded in the notebook (deep trees, overfits: train R2 0.95).
    "notebook": dict(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    # A more regularised starting point to tune from (not the notebook's submission).
    "regularised": dict(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, random_state=RANDOM_STATE, n_jobs=-1,
    ),
}
