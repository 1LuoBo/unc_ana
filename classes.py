import pandas as pd
from ccbc import SystematicsOutput
from dataclasses import dataclass
import numpy as np

__all__ = ['Uncertainty']

@dataclass
class Uncertainty:
    bins: np.ndarray
    out: SystematicsOutput

    @property
    def frac_bin(self):
        return np.sqrt(np.diag(self.out.rate_cov)) / self.out.rate_hist_cv
    @property
    def frac_total(self):
        return np.sqrt(np.sum(self.out.rate_cov)) / np.sum(self.out.rate_hist_cv)
    def knob(self, key):
        return self.out.rate_syst_df.query('key == @key').unc_diag.iloc[0]