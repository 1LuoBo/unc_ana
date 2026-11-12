import numpy as np
import pandas as pd
import pytest

from unc_ana import (compute_mode_hists, breakdown, breakdown_groups,
                     check_closure, compare_samples)

BINS = np.array([0.5, 1, 1.7, 2.5, 3.5, 5, 7])
KNOBS = {"GENIE_a": 100, "GENIE_b": 1, "Flux_c": 100}   # knob -> nuniv (1 == unisim)


def make_df(n=20000, seed=0, mode_probs=(0.3, 0.3, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    d = {"E": rng.exponential(1.5, n) + 0.5,
         "genie_mode": rng.choice([0, 1, 2, 10], n, p=mode_probs),
         "weights_mc": np.ones(n)}
    mode = d["genie_mode"]
    for k, nu in KNOBS.items():
        # mode-dependent spread so channels genuinely differ
        spread = np.where(mode == 0, 0.3, 0.1)
        d.update({f"{k}__u{i}": 1 + spread * rng.normal(size=n) for i in range(nu)})
    return pd.DataFrame(d)


def fake_hist_fn(df, var, bins, mcbnb_pot=None, **_):
    """numpy re-implementation of the nueana.get_syst_hists contract."""
    idx = np.clip(np.searchsorted(bins, df[var].to_numpy(), side="right") - 1, 0, len(bins) - 2)
    w0 = df.weights_mc.to_numpy()
    n = len(bins) - 1
    cv = np.bincount(idx, weights=w0, minlength=n).astype(float)
    out = {}
    for k, nu in KNOBS.items():
        W = df[[f"{k}__u{i}" for i in range(nu)]].to_numpy() * w0[:, None]
        out[k] = {"hists": np.array([np.bincount(idx, weights=W[:, u], minlength=n)
                                     for u in range(nu)]).T}
    return out, cv


@pytest.fixture(scope="module")
def tidy():
    mh = compute_mode_hists(make_df(), "E", BINS, mode_col="genie_mode", hist_fn=fake_hist_fn)
    return breakdown(mh)


def test_closure_exact(tidy):
    assert check_closure(tidy, rtol=1e-10) < 1e-10


def test_total_matches_unsplit(tidy):
    """var_total equals the covariance diagonal computed on the unsplit sample."""
    df = make_df()
    for k in KNOBS:
        syst, cv = fake_hist_fn(df, "E", BINS)
        D = syst[k]["hists"] - cv[:, None]
        expect = (D ** 2).sum(1) / D.shape[1]
        got = tidy[tidy.knob == k].groupby("bin").var_total.first().to_numpy()
        np.testing.assert_allclose(got, expect, rtol=1e-10)


def test_cv_partition(tidy):
    cvm = tidy[tidy.knob == "GENIE_a"].groupby("bin").cv_mode.sum().to_numpy()
    cvt = tidy[tidy.knob == "GENIE_a"].groupby("bin").cv_total.first().to_numpy()
    np.testing.assert_allclose(cvm, cvt)


def test_intrinsic_reflects_input_spread(tidy):
    """QE was generated with 3x the weight spread; its intrinsic frac should be larger."""
    t = tidy[(tidy.knob == "GENIE_a") & (tidy.bin == 0)].set_index("mode")
    assert t.loc[0, "frac_mode"] > 2 * t.loc[1, "frac_mode"]


def test_group_sum_is_additive(tidy):
    g = breakdown_groups(tidy)
    gen = g[(g.group == "GENIE") & (g["mode"] == 0)].sort_values("bin")
    manual = tidy[tidy.group == "GENIE"].query("mode == 0").groupby("bin").var_mode.sum()
    np.testing.assert_allclose(gen.var_mode.to_numpy(), manual.to_numpy())
    assert set(g.group) == {"GENIE", "Flux"}


def test_group_filter():
    mh = compute_mode_hists(make_df(), "E", BINS, mode_col="genie_mode",
                            hist_fn=fake_hist_fn, group="Flux")
    assert list(mh.univ) == ["Flux_c"]


def test_compare_samples_with_different_channel_mix():
    a = breakdown(compute_mode_hists(make_df(seed=1), "E", BINS, mode_col="genie_mode", hist_fn=fake_hist_fn))
    b = breakdown(compute_mode_hists(make_df(seed=2, mode_probs=(0.1, 0.4, 0.2, 0.3)), "E", BINS,
                                     mode_col="genie_mode", hist_fn=fake_hist_fn))
    m = compare_samples(a, b, level="group")
    assert {"ratio_frac_mode", "frac_contrib_mu", "frac_contrib_e"} <= set(m.columns)
    # same physics per channel -> intrinsic ratio ~1 even though the mix differs
    assert abs(np.nanmedian(m.ratio_frac_mode) - 1) < 0.1
    # ... while the whole-sample uncertainty differs because of the mix
    assert not np.allclose(m.frac_total_mu, m.frac_total_e, rtol=0.01)
