import seaborn as sns
import matplotlib.colors as mcolors


__all__ = ['genie_dict', 'color_mode', 'pdg', 'color_pdg', 'pdg_ls']

def _lighten(color, amount=0.5):
    """Blend a colour toward white (amount=0 -> unchanged, 1 -> white)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

genie_dict = {0:'QE', 1:'RES', 2:'DIS', 3:'COH', 10:'MEC'}
_mode_palette = sns.color_palette("Dark2", n_colors=7)
color_mode = {0: _mode_palette[0], 1:_mode_palette[1], 2:_mode_palette[2],
              3: _mode_palette[3], 10:_mode_palette[4]}
color_pdg = {13:'#0072B2', -13:_lighten('#0072B2'), 211:'#009E73', -211:_lighten('#009E73'), 
             321:'#D55E00', -321:_lighten('#D55E00'), 130:'#CC79A7'}
pdg = {13: r'$\mu^-$', -13: r'$\mu^+$', 211:r'$\pi^+$', 321:r'$K^+$', -321:r'$K^-$', 130:r'$K_L^0$', -211:r'$\pi^-$'}
pdg_ls = {13:'-', -13:'--', 211:'-', -211:'--', 321:'-', -321:'--', 130:'-'}