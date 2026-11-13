# unc_ana

Uncertainty breakdown by interaction channel vs. neutrino energy for the SBND
nu_mu -> nu_e constraint study. Built on the `ccbc` branches of
[`lynnt20/nueana`](https://github.com/lynnt20/nueana) (a dependency; not vendored here).

## Install
```bash
pip install -e ".[dev]"      # nueana must be importable (PYTHONPATH) for real data
pytest
```

## Usage
```python
import numpy as np, unc_ana as ua

ua.add_kinematics(numu_df, nue_df)                 # q0, modq, lepton_genE
w_mu, w_e = ua.reweight_two_way(numu_df, nue_df)   # features default to [('E','','')]
numu_df["weights_mc"], nue_df["weights_mc"] = w_mu, w_e
ua.test_classifier(numu_df, nue_df, features=[("q0", "", ""), ("modq", "", "")])

fracs = ua.calc_frac_df(numu_df, nue_df, pot_numu, pot_nue)       # sector="Flux" for flux only
ua.scatter_fracs(fracs)
frac_mu, frac_e = ua.get_cov_bins(numu_df, nue_df, [0.5, 1, 1.7, 2.5, 3.5, 5, 7], pot_numu, pot_nue)
ua.hist_step(numu_df.E, nue_df.E, np.linspace(0.5, 5, 10), r"$E_\nu$",
             weights_mu=numu_df.weights_mc, weights_e=nue_df.weights_mc)
```

Two-way reweighting: nu_mu is reweighted to nu_e, then nu_e to the *reweighted* nu_mu, so
neither sample keeps phase space the other lacks. For a per-interaction-channel breakdown,
call `calc_frac_df` on `df[df.genie_mode == m]` subsets (`bins` is a keyword argument).

