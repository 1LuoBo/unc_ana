"""Derived kinematic columns used throughout the notebook."""
import numpy as np


__all__ = ['add_kinematics']

def add_kinematics(numu_df, nue_df):
    """Add ``q0``, ``modq`` and ``lepton_genE`` columns in place (as in the notebook)."""
    for df, lepton in ((numu_df, numu_df.mu.genE), (nue_df, nue_df.e.genE)):
        df["q0"] = df.E - lepton
        df["modq"] = np.sqrt(df.Q2 + df["q0"] ** 2)
        df["lepton_genE"] = lepton
    return numu_df, nue_df
