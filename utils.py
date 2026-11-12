import pandas as pd
import ccbc
import numpy as np


def sort_part(df, by_col, ascending = False):
    first_row = df.iloc[:1]
    rest_df = df.iloc[1:]
    rest_df = rest_df.sort_values(by = by_col, ascending = ascending)
    return pd.concat([first_row, rest_df])

def calc_frac_df(numu_df, nue_df):
    systoutput_mu = ccbc.get_total_cov(reco_df = numu_df, 
                                          reco_var = ('mu', 'genE', ''),
                                          bins = [0.5, 7],
                                          mcbnb_pot = pot_numu, 
                                          uncertainty_keys = ['rate', 'norm', 'cosmic'])
    systoutput_e = ccbc.get_total_cov(reco_df = nue_df,
                                 reco_var = ('e', 'genE', ''),
                                 bins = [0.5, 7],
                                 mcbnb_pot = pot_nue,
                                 uncertainty_keys = ['rate', 'norm', 'cosmic'])
    global universes
    universes = list(systoutput_mu.rate_syst_dict.keys())
    fracs_df = pd.DataFrame(columns = ['knob', 'cov_mu', 'frac_mu', 'cov_e', 'frac_e'])
    cov_mu = systoutput_mu.rate_cov[0][0]
    frac_mu = np.sqrt(cov_mu) / systoutput_mu.rate_hist_cv[0]
    cov_e = systoutput_e.rate_cov[0][0]
    frac_e = np.sqrt(cov_e) / systoutput_e.rate_hist_cv[0]
    fracs_df.loc[0] = {'knob': 'Total', 
                       'cov_mu': cov_mu,
                       'frac_mu': frac_mu,
                       'cov_e': cov_e,
                       'frac_e': frac_e,
                      }
    for uni in universes:
        cov_mu = systoutput_mu.rate_syst_dict[uni]['cov'][0][0]
        cov_e = systoutput_e.rate_syst_dict[uni]['cov'][0][0]
        frac_mu = np.sqrt(cov_mu) / systoutput_mu.rate_hist_cv[0]
        frac_e = np.sqrt(cov_e) / systoutput_e.rate_hist_cv[0]
        if ("GENIE" in uni):
            uni_name = uni[21:]
            if ('multi' in uni_name):
                uni_name = uni_name[uni_name.find("_") + 1:]
        else:
            uni_name = uni[uni.find('SBN_v3_') + 7:]
        if frac_e == 0:
            new_row = {'knob': uni_name, 
                       'cov_mu': cov_mu, 
                       'frac_mu': frac_mu,
                       'cov_e': cov_e,
                       'frac_e': frac_e,
                      }
        else:
            diff_mu_e = (frac_mu - frac_e) / frac_e
            new_row = {'knob': uni_name, 
                       'cov_mu': cov_mu, 
                       'frac_mu': frac_mu,
                       'cov_e': cov_e,
                       'frac_e': frac_e,
                      }
        fracs_df.loc[len(fracs_df)] = new_row
    
    fracs_df = fracs_df.set_index('knob')
    fracs_df = sort_part(fracs_df, 'frac_mu')
    return fracs_df