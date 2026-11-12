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

bins = np.array([0.5, 1, 1.7, 2.5, 3.5, 5, 7])
mh_mu = ua.compute_mode_hists(numu_df, ("E", "", ""), bins, mode_col=("genie_mode", "", ""))
mh_e  = ua.compute_mode_hists(nue_df,  ("E", "", ""), bins, mode_col=("genie_mode", "", ""))
t_mu, t_e = ua.breakdown(mh_mu), ua.breakdown(mh_e)
ua.check_closure(t_mu); ua.check_closure(t_e)      # channels sum to the total
cmp = ua.compare_samples(t_mu, t_e, level="group")  # or level="knob"
```

`frac_mode` is a channel's own uncertainty (compare nu_mu vs nu_e at fixed channel);
`frac_contrib` is its additive share of the sample uncertainty (this is where a
different channel mix shows up). See the module docstring of `uncertainty.py`.

## Status
`utils.py` is the earlier notebook-derived draft and is superseded by `src/unc_ana/`.
