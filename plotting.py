"""Plot helpers from the analysis notebook."""
import matplotlib.pyplot as plt
import numpy as np
from .constants import *
from .uncertainty import *
from .classes import *

__all__ = ['scatter_fracs', 'hist_step', 'q0q3', 
           'unc_plot_by_mode']



def scatter_fracs(fracs_df):
    """frac_mu vs frac_e per knob (top) and their difference (bottom)."""
    fig, axs = plt.subplots(2, 1, sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    knobs = fracs_df.iloc[1:]
    knobs = knobs[knobs.frac_e > 0]
    boundary = max(fracs_df.iloc[1:].frac_mu.max(), fracs_df.iloc[1:].frac_e.max())
    ax = axs[0]
    ax.scatter(knobs.frac_e, knobs.frac_mu)
    ax.plot([0, boundary], [0, boundary], color="red", label="frac_mu = frac_e")
    ax.set_ylabel("frac_mu")
    ax.legend()
    ax = axs[1]
    ax.scatter(knobs.frac_e, knobs.frac_mu - knobs.frac_e)
    ax.plot([0, boundary], [0, 0], color="red")
    ax.set_ylabel("frac_mu - frac_e")
    ax.set_xlabel("frac_e")
    return fig


def hist_step(numu_dat, nue_dat, bins, xlabel, title="Area Normalized", vlines=None,
              weights_mu=None, weights_e=None):
    """Area-normalized nu_mu vs nu_e histogram with a ratio panel. Overflow goes in the last bin."""
    bins = np.asarray(bins)
    if weights_mu is None:
        weights_mu = np.ones(len(numu_dat))
    if weights_e is None:
        weights_e = np.ones(len(nue_dat))
    weights_mu, weights_e = np.asarray(weights_mu), np.asarray(weights_e)
    fig, axs = plt.subplots(2, 1, sharex="col", gridspec_kw={"height_ratios": [3, 1]})

    ax = axs[0]
    a_max = np.average([bins[-1], bins[-2]])
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    a = ax.hist(np.clip(numu_dat, a_min=bins[0], a_max=a_max), density=True, color="orange",
                label="numu", alpha=0.7, bins=bins, weights=weights_mu)
    b = ax.hist(np.clip(nue_dat, a_min=bins[0], a_max=a_max), density=True, color="blue",
                label="nue", alpha=0.7, bins=bins, weights=weights_e)
    ax.legend()
    ymax = max(max(a[0]), max(b[0]))
    if vlines is not None:
        ax.vlines(vlines[1:-1], ymin=0, ymax=ymax, color="gray", linestyles="dashed")
    ax.set_title(title)
    err_mu = np.sqrt(a[0]) / np.sqrt((bins[1:] - bins[:-1]) * weights_mu.sum())
    err_e = np.sqrt(b[0]) / np.sqrt((bins[1:] - bins[:-1]) * weights_e.sum())
    ax.errorbar(bin_centers, b[0], yerr=err_e, marker=".", drawstyle="steps-mid",
                linestyle="none", alpha=0.7)
    ax.errorbar(bin_centers, a[0], yerr=err_mu, marker=".", drawstyle="steps-mid",
                linestyle="none", alpha=0.7)

    ax = axs[1]
    ratio = np.where(b[0] != 0, a[0] / b[0], 1)
    ax.step(bins, np.append(ratio[0], ratio))
    ax.hlines(y=1, xmin=bins[0], xmax=bins[-1], color="gray", linestyles="dashed")
    unc_ratio = ratio * np.sqrt(np.where(a[0] != 0, err_mu / a[0], 0) ** 2
                                + np.where(b[0] != 0, err_e / b[0], 0) ** 2)
    ax.fill_between(bins, np.append(ratio + unc_ratio, ratio[-1] + unc_ratio[-1]),
                    np.append(ratio - unc_ratio, ratio[-1] - unc_ratio[-1]),
                    step="post", color="lightgray", alpha=0.6)
    ax.set_ylabel("ratio")
    ax.set_xlabel(xlabel)
    return fig


def q0q3(numu_df, nue_df, nbins=20, q0_bins=None):
    """2D area-normalized nu_mu/nu_e ratio map in (q3, q0); ``q0_bins`` draws dashed bin lines."""
    import scipy.ndimage
    q0_edges = np.linspace(0, 1, nbins + 1)
    q3_edges = np.linspace(0, 2, nbins + 1)
    hist_numu, _, _ = np.histogram2d(numu_df["modq"].values, numu_df["q0"].values,
                                     bins=[q3_edges, q0_edges],
                                     weights=numu_df["weights_mc"].values, density=True)
    hist_nue, _, _ = np.histogram2d(nue_df["modq"].values, nue_df["q0"].values,
                                    bins=[q3_edges, q0_edges],
                                    weights=nue_df["weights_mc"].values, density=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_map = hist_numu / hist_nue
        ratio_map[np.isnan(ratio_map)] = 1
        ratio_map[np.isinf(ratio_map)] = 1
    ratio_map = scipy.ndimage.gaussian_filter(ratio_map, sigma=0.8)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(q3_edges, q0_edges, ratio_map.T, cmap="coolwarm", vmin=0.5, vmax=1.5,
                       shading="auto")
    if q0_bins is not None:
        ax.hlines(np.asarray(q0_bins)[1:-1], xmin=0, xmax=2, color="gray", linestyles="dashed")
    fig.colorbar(im, ax=ax).set_label(r"Ratio ($\nu_\mu$ / $\nu_e$)", fontsize=12)
    ax.set_xlabel("$q_3$ (GeV)", fontsize=12)
    ax.set_ylabel("$q_0$ (GeV)", fontsize=12)
    ax.set_title("2D Area-Normalized Ratio Map in $q_0 - q_3$ Space", fontsize=14)
    fig.tight_layout()
    return fig

def unc_plot_by_mode(df, bins, pot, var = ('E', '', '')):
    print('Running through all interaction modes.')
    out_tot = get_cov_bins(df, bins, pot, var)
    outs = {}
    for i in genie_dict.keys():
        print('Running through '+ genie_dict[i])
        df_int = df[df.genie_mode == i]
        outs[i] = get_cov_bins(df_int, bins, pot, var)
    plt.stairs(values = out_tot.frac_bin, 
               edges = bins, 
               label = f'Total ({100 * out_tot.frac_total:.2e}%)',
               color = 'black')
    for i in genie_dict.keys():        
        plt.stairs(values = outs[i].frac_bin, 
                   edges = bins, 
                   color = color_mode[i],
                   label = f'Total ({100 * outs[i].frac_total:.2e}%)')
        plt.legend()