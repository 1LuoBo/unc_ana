"""Per-interaction-channel uncertainty breakdown as a function of a binned variable.

Background
----------
``nueana.syst.get_syst_hists`` returns, for every systematic knob, the universe
histograms ``hists`` with shape ``(nbins, nuniv)`` (``nuniv == 1`` for unisims)
plus the CV histogram.  ``nueana`` then builds the covariance per knob as

    cov = D @ D.T / nuniv,   D = hists - cv[:, None]

Because every universe weight is a per-event number, the universe histogram of
the *whole* sample is the sum over interaction channels of the channel
histograms, hence also  D_total = sum_m D_m.  This gives an exact, additive
decomposition of the total variance of each knob:

    var_total[b] = sum_m  Cov(h_m, h_total)[b,b]  =  sum_m (D_m * D_total).sum(1)/nuniv

(cross-channel correlations are kept, assigned half to each channel involved,
so the channel contributions add up to the total with no remainder).  The
"intrinsic" uncertainty of a channel, ``sqrt((D_m**2).sum(1)/nuniv) / cv_m``,
is the quantity to compare between nu_mu and nu_e at fixed channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# GENIE scattering-type codes, as used in nueana.analysis.mode_categories
MODE_LABELS = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 10: "MEC"}

# Ordered (first match wins) substring -> group.  Mirrors nueana.syst categories.
GROUP_PATTERNS = (
    ("Flux", "Flux"),
    ("Geant4", "Geant4"),
    ("MCstat", "MCstat"),
    ("GENIE", "GENIE"),
    ("SBNNuSyst", "GENIE"),
    ("SuSAv2", "GENIE"),
)


def classify_knob(key: str, patterns=GROUP_PATTERNS) -> str:
    """Map a raw knob name to a high-level group (``'Other'`` if unknown)."""
    for pat, group in patterns:
        if pat in key:
            return group
    return "Other"


def _default_hist_fn() -> Callable:
    """Locate nueana's ``get_syst_hists`` (package layout or the flat ``ccbc`` copy)."""
    try:
        from nueana.syst import get_syst_hists
        return get_syst_hists
    except ImportError:
        import ccbc  # flat-layout copy used on the analysis machines
        return ccbc.get_syst_hists


@dataclass
class ModeHists:
    """CV and universe histograms split by interaction channel.

    Attributes
    ----------
    bins : bin edges
    cv : {mode: (nbins,)}
    univ : {knob: {mode: (nbins, nuniv)}}
    """

    bins: np.ndarray
    cv: dict
    univ: dict = field(default_factory=dict)

    @property
    def modes(self) -> list:
        return list(self.cv)

    @property
    def cv_total(self) -> np.ndarray:
        return np.sum([self.cv[m] for m in self.cv], axis=0)


def compute_mode_hists(
    df: pd.DataFrame,
    var,
    bins: Sequence[float],
    mode_col=("genie_mode", "", ""),
    mcbnb_pot: float | None = None,
    hist_fn: Callable | None = None,
    group: str | Iterable[str] | None = None,
) -> ModeHists:
    """Run ``get_syst_hists`` on each interaction-channel subset of ``df``.

    ``df`` must carry the multiverse weights (``make_multiverse_weights``) and
    ``weights_mc``.  Every value of ``mode_col`` present in ``df`` becomes a
    channel, so the channels partition the sample and closure holds exactly.
    ``group`` optionally restricts to knob groups (e.g. ``'Flux'``).
    """
    bins = np.asarray(bins, dtype=float)
    hist_fn = hist_fn or _default_hist_fn()
    keep = None if group is None else ({group} if isinstance(group, str) else set(group))

    out = ModeHists(bins=bins, cv={})
    mode_vals = df[mode_col].to_numpy()
    for m in sorted(pd.unique(mode_vals)):
        sub = df[mode_vals == m]
        syst, cv = hist_fn(sub, var, bins, mcbnb_pot=mcbnb_pot)
        out.cv[m] = np.asarray(cv, dtype=float)
        for knob, entry in syst.items():
            if keep is not None and classify_knob(knob) not in keep:
                continue
            out.univ.setdefault(knob, {})[m] = np.asarray(entry["hists"], dtype=float)
    return out


def _deviations(mh: ModeHists, knob: str) -> dict:
    """{mode: D_m (nbins, nuniv)}; a channel absent for this knob contributes 0."""
    per_mode = mh.univ[knob]
    nuniv = next(iter(per_mode.values())).shape[1]
    return {
        m: (per_mode[m] - mh.cv[m][:, None]) if m in per_mode else np.zeros((len(mh.cv[m]), nuniv))
        for m in mh.cv
    }


def breakdown(mh: ModeHists, mode_labels: dict | None = None) -> pd.DataFrame:
    """Tidy table, one row per (knob, mode, bin).

    Columns
    -------
    knob, group, mode, mode_label, bin, bin_lo, bin_hi
    cv_mode, cv_total          : CV counts of the channel / the whole sample
    var_mode                   : variance of the channel histogram alone
    var_contrib                : channel's additive share of the knob's total variance
                                 (sums over modes to ``var_total``)
    var_total                  : knob variance of the whole sample
    frac_mode                  : sqrt(var_mode)/cv_mode    (intrinsic channel uncertainty)
    frac_total                 : sqrt(var_total)/cv_total  (uncertainty of the sample)
    frac_contrib               : var_contrib / cv_total**2 (adds in variance across modes)
    """
    labels = {**MODE_LABELS, **(mode_labels or {})}
    nb = len(mh.bins) - 1
    cv_tot = mh.cv_total
    frames = []
    for knob in mh.univ:
        D = _deviations(mh, knob)
        nuniv = next(iter(D.values())).shape[1]
        D_tot = sum(D.values())
        var_tot = (D_tot ** 2).sum(1) / nuniv
        for m, Dm in D.items():
            var_m = (Dm ** 2).sum(1) / nuniv
            contrib = (Dm * D_tot).sum(1) / nuniv
            cvm = mh.cv[m]
            with np.errstate(divide="ignore", invalid="ignore"):
                frames.append(pd.DataFrame({
                    "knob": knob, "group": classify_knob(knob),
                    "mode": m, "mode_label": labels.get(m, f"mode{m}"),
                    "bin": np.arange(nb), "bin_lo": mh.bins[:-1], "bin_hi": mh.bins[1:],
                    "cv_mode": cvm, "cv_total": cv_tot,
                    "var_mode": var_m, "var_contrib": contrib, "var_total": var_tot,
                    "frac_mode": np.where(cvm > 0, np.sqrt(var_m) / cvm, np.nan),
                    "frac_total": np.where(cv_tot > 0, np.sqrt(var_tot) / cv_tot, np.nan),
                    "frac_contrib": np.where(cv_tot > 0, contrib / cv_tot ** 2, np.nan),
                }))
    return pd.concat(frames, ignore_index=True)


def breakdown_groups(tidy: pd.DataFrame) -> pd.DataFrame:
    """Sum knob variances within each (group, mode, bin), as ``get_total_cov`` does.

    Knobs are treated as independent, so variances (not fractions) are added.
    """
    keys = ["group", "mode", "mode_label", "bin", "bin_lo", "bin_hi", "cv_mode", "cv_total"]
    g = tidy.groupby(keys, as_index=False)[["var_mode", "var_contrib", "var_total"]].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        g["frac_mode"] = np.where(g.cv_mode > 0, np.sqrt(g.var_mode) / g.cv_mode, np.nan)
        g["frac_total"] = np.where(g.cv_total > 0, np.sqrt(g.var_total) / g.cv_total, np.nan)
        g["frac_contrib"] = np.where(g.cv_total > 0, g.var_contrib / g.cv_total ** 2, np.nan)
    return g


def check_closure(tidy: pd.DataFrame, rtol: float = 1e-8) -> float:
    """Max relative deviation of ``sum_m var_contrib`` from ``var_total``; raises if > rtol."""
    s = tidy.groupby(["knob", "bin"]).agg(c=("var_contrib", "sum"), t=("var_total", "first"))
    scale = np.maximum(np.abs(s.t), np.finfo(float).tiny)
    worst = float((np.abs(s.c - s.t) / scale).max())
    if worst > rtol:
        raise AssertionError(f"channel contributions do not sum to total (max rel dev {worst:.2e})")
    return worst


def compare_samples(tidy_mu: pd.DataFrame, tidy_e: pd.DataFrame, level: str = "group") -> pd.DataFrame:
    """Join nu_mu and nu_e tables for side-by-side comparison / ratios.

    ``level='group'`` aggregates knobs first; ``level='knob'`` keeps individual knobs.
    """
    if level == "group":
        a, b, key = breakdown_groups(tidy_mu), breakdown_groups(tidy_e), "group"
    else:
        a, b, key = tidy_mu, tidy_e, "knob"
    on = [key, "mode_label", "bin", "bin_lo", "bin_hi"]
    cols = ["frac_mode", "frac_total", "frac_contrib", "cv_mode", "cv_total"]
    m = a[on + cols].merge(b[on + cols], on=on, suffixes=("_mu", "_e"))
    with np.errstate(divide="ignore", invalid="ignore"):
        m["ratio_frac_mode"] = m.frac_mode_mu / m.frac_mode_e
    return m
