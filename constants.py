import seaborn as sns

__all__ = ['genie_dict', 'color_mode', 'pdg']

genie_dict = {0:'QE', 1:'RES', 2:'DIS', 3:'COH', 10:'MEC'}
_mode_palette = sns.color_palette("Dark2", n_colors=7)
color_mode = {0: _mode_palette[0], 1:_mode_palette[1], 2:_mode_palette[2],
              3: _mode_palette[3], 10:_mode_palette[4]}
pdg = {13: r'$\mu^-$', -13: r'$\mu^+$', 211:r'$\pi$', 321:r'$K^+$', -321:r'$K^-$', 130:r'$K_L^0$'}