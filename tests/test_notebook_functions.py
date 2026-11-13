from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from unc_ana import (calc_frac_df, get_cov_bins, hist_step, scatter_fracs, q0q3,
                     reweight_two_way, add_kinematics)


def fake_total_cov(reco_df, reco_var, bins, mcbnb_pot, uncertainty_keys):
    """Stands in for ccbc.get_total_cov; scale makes nu_mu knobs 2x larger than nu_e."""
    scale = 2.0 if reco_df.attrs["flavor"] == "mu" else 1.0
    n = len(bins) - 1
    cv = np.full(n, 100.0)
    knobs = {"GENIE_xx_multisim_ZExpA1": 4.0, "GENIE_xx_multisigma_MaCCQE": 9.0,
             "FluxKaon_": 1.0, "SBN_v3_foo_Bar": 0.25}
    d = {k: {"cov": np.eye(n) * (v * scale) ** 2} for k, v in knobs.items()}
    total = sum(e["cov"] for e in d.values())
    return SimpleNamespace(rate_hist_cv=cv, rate_cov=total, rate_syst_dict=d)


def dfs():
    mu = pd.DataFrame({"E": [1.0]}); mu.attrs["flavor"] = "mu"
    e = pd.DataFrame({"E": [1.0]}); e.attrs["flavor"] = "e"
    return mu, e


def test_calc_frac_df_total_first_and_sorted():
    mu, e = dfs()
    f = calc_frac_df(mu, e, 1e20, 1e20, total_cov_fn=fake_total_cov)
    assert f.index[0] == "Total"
    rest = f.frac_mu.iloc[1:].to_numpy()
    assert np.all(np.diff(rest) <= 0)                       # sorted descending
    np.testing.assert_allclose(f.frac_mu.iloc[1:] / f.frac_e.iloc[1:], 2.0)
    np.testing.assert_allclose(f.loc["Total", "cov_mu"], sum((v * 2) ** 2 for v in (4, 9, 1, .25)))


def test_calc_frac_df_flux_sector_only():
    mu, e = dfs()
    f = calc_frac_df(mu, e, 1e20, 1e20, sector="Flux", total_cov_fn=fake_total_cov)
    assert len(f) == 2 and f.index[0] == "Total"


def test_get_cov_bins_shapes():
    mu, e = dfs()
    bins = [0.5, 1, 1.7, 2.5]
    fm, fe = get_cov_bins(mu, e, bins, 1e20, 1e20, total_cov_fn=fake_total_cov)
    assert fm.shape == fe.shape == (3,)
    np.testing.assert_allclose(fm / fe, 2.0)


def make_samples(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    # nu_mu is softer and extends lower; nu_e is harder. Overlapping but different support.
    mu = pd.DataFrame({"E": rng.gamma(2.0, 0.8, n) + 0.3})
    e = pd.DataFrame({"E": rng.gamma(3.5, 0.7, n) + 0.5})
    mu["weights_mc"] = 1.0
    e["weights_mc"] = 1.0
    return mu, e


def _wmean(x, w):
    return np.average(x, weights=w)


def test_reweight_two_way_matches_distributions_and_does_not_mutate():
    mu, e = make_samples()
    feats = ["E"]
    w_mu, w_e = reweight_two_way(mu, e, features=feats, min_samples_leaf=100)
    assert (mu.weights_mc == 1).all() and (e.weights_mc == 1).all()    # inputs untouched
    bins = np.linspace(0.5, 6, 12)
    h_mu = np.histogram(mu.E, bins, weights=w_mu, density=True)[0]
    h_e = np.histogram(e.E, bins, weights=w_e, density=True)[0]
    h_mu0 = np.histogram(mu.E, bins, density=True)[0]
    h_e0 = np.histogram(e.E, bins, density=True)[0]
    # the reweighted shapes agree far better than the originals
    assert np.abs(h_mu - h_e).max() < 0.5 * np.abs(h_mu0 - h_e0).max()
    assert abs(_wmean(mu.E, w_mu) - _wmean(e.E, w_e)) < 0.1 * abs(mu.E.mean() - e.E.mean())


def test_reweight_two_way_returns_weights_aligned_with_rows():
    mu, e = make_samples(n=2000, seed=3)
    mu.index = np.arange(len(mu)) + 500        # non-default index must not break alignment
    w_mu, w_e = reweight_two_way(mu, e, features=["E"], min_samples_leaf=100)
    assert w_mu.shape == (len(mu),) and w_e.shape == (len(e),)
    assert np.all(w_mu >= 0) and np.all(w_e >= 0)


def test_plots_smoke():
    mu, e = make_samples(n=2000)
    fig = hist_step(mu.E, e.E, bins=np.linspace(0.5, 5, 10), xlabel="E",
                    weights_mu=mu.weights_mc, weights_e=e.weights_mc)
    assert fig is not None
    f = pd.DataFrame({"frac_mu": [.3, .2, .1], "frac_e": [.25, .1, 0.0]},
                     index=["Total", "a", "b"])
    scatter_fracs(f)
    for d in (mu, e):
        d["q0"] = d.E * 0.3
        d["modq"] = d.E * 0.5
    q0q3(mu, e, 10, q0_bins=[0, .05, .1, .2, .7, 1])


def test_add_kinematics():
    cols = pd.MultiIndex.from_tuples([("E", "", ""), ("Q2", "", ""), ("mu", "genE", "")])
    df = pd.DataFrame([[2.0, 0.5, 1.0]], columns=cols)
    cols_e = pd.MultiIndex.from_tuples([("E", "", ""), ("Q2", "", ""), ("e", "genE", "")])
    de = pd.DataFrame([[3.0, 0.5, 1.0]], columns=cols_e)
    add_kinematics(df, de)
    assert df["q0"].iloc[0] == 1.0 and de["q0"].iloc[0] == 2.0
    np.testing.assert_allclose(df["modq"].iloc[0], np.sqrt(0.5 + 1.0))
