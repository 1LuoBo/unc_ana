"""Plots for the per-channel uncertainty breakdown (inputs come from ``uncertainty``)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .uncertainty import breakdown_groups, compare_samples


def _step(ax, lo, hi, y, **kw):
    edges = np.append(lo, hi[-1])
    ax.step(edges, np.append(y, y[-1]), where="post", **kw)


def plot_intrinsic_vs_energy(tidy_mu, tidy_e, group="GENIE", xlabel=r"$E_\nu$ (GeV)"):
    """Per channel: intrinsic fractional uncertainty, nu_mu (solid) vs nu_e (dashed), with ratio."""
    m = compare_samples(tidy_mu, tidy_e, level="group")
    m = m[m.group == group]
    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i, (lab, d) in enumerate(m.groupby("mode_label")):
        d = d.sort_values("bin")
        c = f"C{i}"
        _step(ax, d.bin_lo.to_numpy(), d.bin_hi.to_numpy(), d.frac_mode_mu.to_numpy(), color=c, label=f"{lab} $\\nu_\\mu$")
        _step(ax, d.bin_lo.to_numpy(), d.bin_hi.to_numpy(), d.frac_mode_e.to_numpy(), color=c, ls="--")
        _step(rax, d.bin_lo.to_numpy(), d.bin_hi.to_numpy(), d.ratio_frac_mode.to_numpy(), color=c)
    ax.set_ylabel(f"{group} frac. unc. within channel")
    ax.legend(ncol=2, fontsize=8, title="solid: $\\nu_\\mu$, dashed: $\\nu_e$")
    rax.axhline(1, color="gray", ls=":")
    rax.set_ylabel(r"$\nu_\mu/\nu_e$")
    rax.set_xlabel(xlabel)
    return fig


def plot_contributions_vs_energy(tidy, group="GENIE", ax=None, title="", xlabel=r"$E_\nu$ (GeV)"):
    """Stacked per-channel contribution to the sample's total fractional variance."""
    g = breakdown_groups(tidy)
    g = g[g.group == group]
    ax = ax or plt.subplots()[1]
    piv = g.pivot(index="bin", columns="mode_label", values="frac_contrib")
    lo = g.drop_duplicates("bin").sort_values("bin")
    centers = 0.5 * (lo.bin_lo.to_numpy() + lo.bin_hi.to_numpy())
    width = lo.bin_hi.to_numpy() - lo.bin_lo.to_numpy()
    bottom = np.zeros(len(piv))
    for lab in piv.columns:
        ax.bar(centers, piv[lab].to_numpy(), width=width, bottom=bottom, label=lab, alpha=0.85)
        bottom += piv[lab].to_numpy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{group} (frac. unc.)$^2$ contribution")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax
