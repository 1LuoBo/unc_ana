import pandas as pd
import sys; sys.path.append("/exp/sbnd/app/users/lynnt/cafpyana")
from makedf.util import *

def contained(df):
    return (df.end.z > 5) & (df.end.z < 495) & (df.end.x > -195) & (df.end.x < 195) & (df.end.y > -195) & (df.end.y < 195)


def cont_exit(df):
    cont = contained(df.pfp.trk)
    return (df[cont], df[~cont])

def muInSpill(df):
    spill_start = 0.1
    spill_end = 2.2
    return (df.slc.barycenterFM.flashTime > spill_start) & (df.slc.barycenterFM.flashTime < spill_end)

def muInScore(df):
    return (df.slc.barycenterFM.score > 0.02)

def in_detector(df):
    in_detector = (df.pfp.trk.end.z > 5) & (df.pfp.trk.end.z < 495) & (df.pfp.trk.end.x > -195) & (df.pfp.trk.end.x < 195) & (df.pfp.trk.end.y > -195) & (df.pfp.trk.end.y < 195)
    return in_detector

def clean_duplicates(group):
    if (len(group) == 2):
        sorted_group = group.sort_values(('pfp', 'dist_to_vertex', '', '', '', ''))
        d_small = sorted_group.iloc[0][('pfp', 'dist_to_vertex')]
        d_large = sorted_group.iloc[1][('pfp', 'dist_to_vertex')]
    if (d_small < 5) and (d_large > 20):
        return sorted_group.head(1)
    return group


def muselect(indf):
    df_dict = {}
    df = indf.copy()
    slice_level = ['entry', '__ntuple', 'rec.slc..index']
    is_large = False
    if 'file_idx' in df.index.names:
        slice_level = ['entry', '__ntuple', 'rec.slc..index', 'file_idx']
        is_large = True

        
    # ** these cuts done already in makedf
    # * require nuscore > 0.5
    # * require not clear cosmic 
    # * require reco vertex in AV
    # * require that there is a primary shower (at least one pfp w/ trackScore < 0.5)
    if not is_large:
        df_dict['preselection'] = df

    # * require that the matched (many-to-many) is inside the beam spill
    df = df[muInSpill(df) & muInScore(df)]
    if not is_large:
        df_dict['flash matching'] = df

    # * require that primary shower > 500 GeV
    # df = df[df.primshw.shw.maxplane_energy > 0.5]
    # df_dict['shower energy'] = df 


    # * require that vertex in FV
    df = df[InFV(df.slc.vertex, inzback = 0, det = 'SBND_nohighyz')]
    if not is_large:
        df_dict['vertex FV'] = df
    
    # * require flash PE > 2000
    df = df[df.slc.barycenterFM.flashPEs * 0.66 > 2000]
    if not is_large:
        df_dict['flash PE'] = df
    
    
    # * require start position to be within 1.2cm of vertex
    # df = df[df.pfp.dist_to_vertex < 1.2]
    # df_dict['dist_to_vertex'] = df

    muon_flag_col = ('pfp', 'is_muon_candidate', '', '', '', '')
    if "is_muon_candidate" not in df.pfp.columns:
        df[('pfp', 'is_muon_candidate', '', '', '', '')] = False
    df.loc[:, [muon_flag_col]] = False
    
    # * require at least one muon candidate
    is_muon_track = (
        (df.pfp.trackScore > 0.6) & 
        (df.pfp.trk.len > 32) & 
        (df.pfp.trk.chi2pid.I2.chi2_muon < 20) & 
        (df.pfp.trk.chi2pid.I2.chi2_muon > 0.001) & 
        (df.pfp.trk.chi2pid.I2.chi2_proton > 90) &
        (df.pfp.dist_to_vertex < 1.2)
    )
    df.loc[is_muon_track, muon_flag_col] = True
    muon_mask = df.groupby(level=slice_level)[[muon_flag_col]].transform('any').iloc[:, 0]
    df = df[muon_mask]
    
    if not is_large:
        df_dict['muon candidate'] = df
    
    # * require nu score > 0.5
    # df = df[df.slc.nu_score > 0.5]
    # df_dict['nu score'] = df
    
    # # * require not LowZ
    # df = df[df.pfp.trk.end.z > 6]
    # df_dict['lowZ'] = df

    # * require that muon candidate contained in one TPC
    track_is_contained_candidate = df[muon_flag_col] & ((in_detector(df) & (
        ((df.pfp.trk.start.x > 0) & (df.pfp.trk.end.x > 0)) | 
        ((df.pfp.trk.start.x < 0) & (df.pfp.trk.end.x < 0))
        )) | ~in_detector(df)) 
    nohighyz = track_is_contained_candidate & ~(
        (df.pfp.trk.end.x > -190) & (df.pfp.trk.end.x < -5) & 
        (df.pfp.trk.end.y > 100)  & (df.pfp.trk.end.y < 200) & 
        (df.pfp.trk.end.z > 250)  & (df.pfp.trk.end.z < 450)
    )
    df.loc[:,[muon_flag_col]] = nohighyz
    slice_has_contained_muon = df.groupby(level=slice_level)[[muon_flag_col]].transform('any').iloc[:, 0]
    df = df[slice_has_contained_muon]
    if not is_large:
        df_dict['muon contained'] = df

    # * require momentum > 488.7GeV/c
    # * If the end of the trajectory is well within the detector, then use rangeP. Otherwise use mcs.
    # * z: 486, x:-200 ~ 200, y: -199 ~ 199
    momentum_selection = df[muon_flag_col] & ((in_detector(df) & (df.pfp.trk.rangeP.p_muon > 0.4887)) | 
                                              (~in_detector(df) & (df.pfp.trk.mcsP.fwdP_muon > 0.4887)))
    df.loc[:, [muon_flag_col]] = momentum_selection
    df_momentum = df[df[muon_flag_col]]
    if not df_momentum.empty:
        df.loc[:,[muon_flag_col]] = False
        best_muon_indices = df_momentum.groupby(level=slice_level)[[('pfp', 'trk', 'len', '', '', '')]].idxmax().iloc[:, 0]
        df.loc[best_muon_indices, muon_flag_col] = True
    momentum_idx = df.groupby(level = slice_level)[[muon_flag_col]].transform('any').iloc[:, 0]
    df = df[momentum_idx]
    df_dict['momentum'] = df


    
    return df_dict