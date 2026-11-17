"""Tests for PAT Multivariate PCA (Hotelling T2 / Q-SPE) and NIPALS PLS."""

from __future__ import annotations

import numpy as np
import pytest

from batch_sentinel.pat import MultivariatePATMonitor, PLSQualityPredictor


def test_pat_pca_detects_fault_and_identifies_dominant_driver() -> None:
    rng = np.random.default_rng(123)
    x_train = rng.normal([3.6, 37.0, 7.15, 0.7], [0.08, 0.12, 0.03, 0.15], size=(100, 4))
    mon = MultivariatePATMonitor(n_components=2, alpha=0.01).fit(
        x_train, feature_names=["DO", "Temp", "pH", "Lactate"]
    )

    x_in_control = rng.normal([3.6, 37.0, 7.15, 0.7], [0.08, 0.12, 0.03, 0.15], size=(40, 4))
    clean_audit = mon.score(x_in_control)
    assert not clean_audit.fault_detected
    assert clean_audit.t2_exceedance_pct <= 7.5

    # Inject strong fault on Lactate (column 3)
    x_fault = x_in_control.copy()
    x_fault[20:, 3] += 3.5
    fault_audit = mon.score(x_fault)
    assert fault_audit.fault_detected
    assert fault_audit.spe_exceedance_pct > 30.0
    assert fault_audit.top_fault_drivers[0][0] == "Lactate"


def test_nipals_pls_recovers_linear_cqa_relationship() -> None:
    rng = np.random.default_rng(456)
    x = rng.normal(0.0, 1.0, size=(120, 5))
    # True CQA depends strongly on x[:, 0] and x[:, 2]
    y = 4.5 + 1.8 * x[:, 0] - 0.9 * x[:, 2] + 0.05 * rng.normal(0.0, 1.0, size=120)

    pls = PLSQualityPredictor(n_components=3).fit(x[:90], y[:90])
    preds = pls.predict(x[90:]).ravel()
    rmse = float(np.sqrt(np.mean((preds - y[90:]) ** 2)))
    assert rmse < 0.15
    assert np.corrcoef(preds, y[90:])[0, 1] == pytest.approx(1.0, abs=0.02)
