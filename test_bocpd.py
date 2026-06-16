"""Verification tests for the BOCPD rewrite."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.bocpd import BOCPD, MultiFeatureBOCPD, BOCPDResult

def test_single_detector():
    print("=" * 60)
    print("TEST 1: Single detector — non-zero changepoint signals")
    print("=" * 60)

    det = BOCPD()
    np.random.seed(42)
    data = np.random.randn(500)
    data[200:250] += 3  # inject a mean shift

    results = det.process_batch(data)
    cp_probs = np.array([r.changepoint_probability for r in results])
    alerts = [r.alert for r in results]
    confidences = [r.confidence for r in results]

    print(f"  Total samples:     {len(results)}")
    print(f"  Mean CP prob:      {cp_probs.mean():.6f}")
    print(f"  Max CP prob:       {cp_probs.max():.6f}")
    print(f"  Alerts fired:      {sum(alerts)}")
    print(f"  Confidence[0]:     {confidences[0]:.4f}")
    print(f"  Confidence[100]:   {confidences[min(100, len(confidences)-1)]:.4f}")
    print(f"  Confidence[-1]:    {confidences[-1]:.4f}")

    assert cp_probs.max() > 0.1, f"Max CP prob too low: {cp_probs.max()}"
    assert len(results) == 500, f"Output length mismatch: {len(results)}"
    print("  PASS")


def test_result_fields():
    print()
    print("=" * 60)
    print("TEST 2: BOCPDResult has new fields")
    print("=" * 60)

    det = BOCPD()
    np.random.seed(42)
    data = np.random.randn(500)
    data[200:250] += 3
    results = det.process_batch(data)

    r = results[250]
    print(f"  changepoint_probability: {r.changepoint_probability:.6f}")
    print(f"  growth_probability:      {r.growth_probability:.6f}")
    print(f"  confidence:              {r.confidence:.4f}")
    print(f"  alert:                   {r.alert}")
    assert hasattr(r, "growth_probability"), "Missing growth_probability"
    assert hasattr(r, "confidence"), "Missing confidence"
    assert abs(r.changepoint_probability + r.growth_probability - 1.0) < 1e-6
    print("  PASS")


def test_warmup_gating():
    print()
    print("=" * 60)
    print("TEST 3: Warm-up gating — no alerts in first 100 bars")
    print("=" * 60)

    det = BOCPD()
    np.random.seed(42)
    # Put a huge shift right at the start to try to trigger early alerts
    data = np.random.randn(500)
    data[10:20] += 10  # Big shift during warm-up

    results = det.process_batch(data)
    warmup_alerts = [r.alert for r in results[:100]]
    print(f"  Alerts in warm-up (bars 0-99): {sum(warmup_alerts)}")
    assert not any(warmup_alerts), f"Got {sum(warmup_alerts)} alerts during warm-up!"
    print("  PASS")


def test_multi_feature():
    print()
    print("=" * 60)
    print("TEST 4: MultiFeatureBOCPD — output shape and columns")
    print("=" * 60)

    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "vol_term_structure_slope": np.random.randn(n),
        "return_autocorr_change": np.random.randn(n),
        "drawdown_velocity": np.random.randn(n),
        "bb_width_roc": np.random.randn(n),
        "hurst_exponent": np.random.randn(n),
    })
    # Inject shifts in 2 features at bar 200
    df.loc[200:250, "vol_term_structure_slope"] += 4
    df.loc[200:250, "return_autocorr_change"] += 4

    mbocpd = MultiFeatureBOCPD()
    output = mbocpd.process_batch(df)

    expected_cols = [
        "bocpd_cp_prob", "bocpd_rl_mean", "bocpd_rl_mode",
        "bocpd_rl_entropy", "bocpd_alert", "bocpd_confidence",
    ]
    for col in expected_cols:
        assert col in output.columns, f"Missing column: {col}"

    assert len(output) == n, f"Output length mismatch: {len(output)} != {n}"
    print(f"  Output shape: {output.shape}")
    print(f"  Columns: {list(output.columns)}")
    print(f"  Alerts: {output['bocpd_alert'].sum()}")
    print(f"  Max CP prob: {output['bocpd_cp_prob'].max():.6f}")
    print(f"  Mean confidence: {output['bocpd_confidence'].mean():.4f}")
    print("  PASS")


if __name__ == "__main__":
    test_single_detector()
    test_result_fields()
    test_warmup_gating()
    test_multi_feature()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
