"""Fractional-uncertainty tables for nu_mu vs nu_e (from the analysis notebook)."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["sort_part", "_knob_name", "calc_frac_df", "get_cov_bins"]



def _default_total_cov():
    """``get_total_cov`` from ccbc (flat copy on the cluster) or the nueana package."""
    try:
        import ccbc
        return ccbc.get_total_cov
    except (ImportError, AttributeError):
        import nueana
        return nueana.get_total_cov


def sort_part(df, by_col, ascending=False):
    """Sort everything except the first row (the 'Total' row)."""
    first_row = df.iloc[:1]
    rest_df = df.iloc[1:].sort_values(by=by_col, ascending=ascending)
    return pd.concat([first_row, rest_df])


def _knob_name(uni):
    """Shorten a raw universe key for display (same rules as the notebook)."""
    if "GENIE" in uni:
        name = uni[21:]
        if "multi" in name:
            name = name[name.find("_") + 1:]
        return name
    if "SBN_v3" in uni:
        return uni[uni.find("SBN_v3_") + 7:]
    if "Flux" in uni:
        return uni[:-5]
    return uni


def _run_total_cov(df, pot, bins, total_cov_fn):
    return total_cov_fn(reco_df=df, reco_var=("E", "", ""), bins=bins,
                        mcbnb_pot=pot, uncertainty_keys=["rate", "cosmic"])


def calc_frac_df(numu_df, nue_df, pot_numu, pot_nue, sector=None, bins=None,
                 total_cov_fn=None):
    """Fractional uncertainty of the first bin, per systematic knob, for nu_mu and nu_e.

    Returns a DataFrame indexed by knob (first row 'Total', rest sorted by ``frac_mu``)
    with columns ``cov_mu, frac_mu, cov_e, frac_e``.  With the default ``bins=[0.5, 7]``
    there is a single bin; with several bins only the first one is reported
    (use :func:`get_cov_bins` for all bins).  ``sector='Flux'`` keeps only flux knobs.
    """
    if bins is None:
        bins = [0.5, 7]
    total_cov_fn = total_cov_fn or _default_total_cov()
    out_mu = _run_total_cov(numu_df, pot_numu, bins, total_cov_fn)
    out_e = _run_total_cov(nue_df, pot_nue, bins, total_cov_fn)

    universes = list(out_mu.rate_syst_dict.keys())
    if sector == "Flux":
        universes = [u for u in universes if "Flux" in u]

    def row(cov_mu, cov_e):
        return {"cov_mu": cov_mu, "frac_mu": np.sqrt(cov_mu) / out_mu.rate_hist_cv[0],
                "cov_e": cov_e, "frac_e": np.sqrt(cov_e) / out_e.rate_hist_cv[0]}

    rows = {"Total": row(out_mu.rate_cov[0][0], out_e.rate_cov[0][0])}
    for uni in universes:
        rows[_knob_name(uni)] = row(out_mu.rate_syst_dict[uni]["cov"][0][0],
                                    out_e.rate_syst_dict[uni]["cov"][0][0])
    fracs_df = pd.DataFrame.from_dict(rows, orient="index",
                                      columns=["cov_mu", "frac_mu", "cov_e", "frac_e"])
    fracs_df.index.name = "knob"
    return sort_part(fracs_df, "frac_mu")


def get_cov_bins(numu_df, nue_df, bins, pot_numu, pot_nue, total_cov_fn=None):
    """Per-bin total fractional uncertainty ``(frac_mu, frac_e)`` for the given E bins."""
    total_cov_fn = total_cov_fn or _default_total_cov()
    out_mu = _run_total_cov(numu_df, pot_numu, bins, total_cov_fn)
    out_e = _run_total_cov(nue_df, pot_nue, bins, total_cov_fn)
    frac_mu = np.sqrt(np.diag(out_mu.rate_cov)) / out_mu.rate_hist_cv
    frac_e = np.sqrt(np.diag(out_e.rate_cov)) / out_e.rate_hist_cv
    return frac_mu, frac_e
