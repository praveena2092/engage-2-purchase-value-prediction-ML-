import numpy as np
import pandas as pd
import pytest

from purchase_value import config as C


def make_frame(n: int, with_target: bool, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the competition's column names (real data can't be redistributed)."""
    rng = np.random.default_rng(seed)
    pick = lambda opts: rng.choice(opts, n)  # noqa: E731
    df = pd.DataFrame({
        "date": pick(["20170105", "20170812", "20171203", "20180301"]),
        "sessionId": np.arange(n), "userId": rng.integers(0, n // 2, n),
        "sessionStart": rng.integers(1_480_000_000, 1_520_000_000, n),
        "sessionNumber": rng.integers(1, 50, n), "totalHits": rng.integers(1, 80, n),
        "pageViews": np.where(rng.random(n) < 0.05, np.nan, rng.integers(1, 60, n)),
        "totals.bounces": np.where(rng.random(n) < 0.5, np.nan, 1.0),
        "new_visits": np.where(rng.random(n) < 0.25, np.nan, 1.0),
        "totals.visits": 1, "gclIdPresent": rng.integers(0, 2, n),
        "browser": pick(["Chrome", "Safari", "Firefox", "Edge", "Opera", "IE", "Other"]),
        "os": pick(["Windows", "Macintosh", "Android", "iOS", "Linux", "Chrome OS", "(not set)"]),
        "deviceType": pick(["desktop", "mobile", "tablet"]),
        "device.isMobile": pick([True, False]), "userChannel": pick(["Direct", "Organic Search", "Social", "Referral"]),
        "geoCluster": pick(["Region_1", "Region_2", "Region_3"]),
        "geoNetwork.networkDomain": pick(["domain1", "domain2", "not available in demo dataset"]),
        "geoNetwork.region": pick(["California", "New York", "(not set)", "not available in demo dataset"]),
        "geoNetwork.city": pick(["San Francisco", "New York", "not available in demo dataset"]),
        "geoNetwork.metro": pick(["San Francisco-Oakland-San Jose CA", "(not set)"]),
        "geoNetwork.continent": pick(["Americas", "Asia", "Europe"]),
        "geoNetwork.subContinent": pick(["Northern America", "Southern Asia", "Western Europe"]),
        "locationCountry": pick(["United States", "India", "Germany"]),
        "trafficSource": pick(["(direct)", "google", "youtube.com"]),
        "trafficSource.medium": pick(["organic", "referral", "(none)"]),
        "trafficSource.referralPath": pick(["/", "/mail/u/0", None]),
        "trafficSource.isTrueDirect": pick([True, None]),
        "trafficSource.adContent": pick([None, None, "Top"]),
        "trafficSource.keyword": pick([None, None, "(not provided)"]),
        "trafficSource.campaign": pick([None, "(not set)", "Data promo"]),
        "trafficSource.adwordsClickInfo.slot": pick([None, "Top"]),
        "trafficSource.adwordsClickInfo.isVideoAd": pick([None, False]),
        "trafficSource.adwordsClickInfo.adNetworkType": pick([None, "Google Search"]),
        "trafficSource.adwordsClickInfo.page": pick([np.nan, 1.0]),
        "socialEngagementType": "Not Socially Engaged", "locationZone": 8, "screenSize": 1.0,
        "device.language": "not available in demo dataset", "browserMajor": "not available in demo dataset",
    })
    if with_target:
        df[C.TARGET] = np.where(rng.random(n) < 0.9, 0, rng.integers(1, 5, n) * 10_000_000)
    return df


@pytest.fixture
def train_df():
    return make_frame(300, True, seed=1)


@pytest.fixture
def test_df():
    return make_frame(80, False, seed=2)
