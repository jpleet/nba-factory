import numpy as np
import pandas as pd 
from bokeh.plotting import ColumnDataSource, figure, output_file, save
from bokeh.models import Span

import sys
sys.path.append('../')
from params import *

def plot(df):

    if verbose:
        print('Making interactive tandem plots')

    for k, season_df in df.groupby('seasons'):
        # limit df
        season_top = season_df[(
            season_df.min_duo_poss > 
            season_df.min_duo_poss.quantile(tandem_quantile))]
        # format season parameter
        if len(k) == 0:
            s = min(pbp_seasons) - 1
            e = max(pbp_seasons) + 1
            k = f'{s}-{e}'
        elif len(k) == 1:
            k = k[0][0:4] + '-' + k[0][4:]
        else:
            k = ', '.join(i[:4]+'-'+i[4:] for i in k)
        # scale marker sizes (relative to top)
        marker_size = (((season_top.min_duo_poss - 
                        season_df.min_duo_poss.min()) / 
                       (season_df.min_duo_poss.max() - 
                        season_df.min_duo_poss.min())) + 0.1) * 10
        # create figure
        sfile = f'{prediction_folder}/tandem_ppp_{k}.html'
        output_file(sfile)

        source = ColumnDataSource(data=dict(
            x = season_top.off_wmn.values, 
            y = season_top.def_wmn.values, 
            name1 = season_top.name1.values,
            name2 = season_top.name2.values,
            posses = season_top.min_duo_poss.values, 
            mark_size = marker_size))   

        tooltip = [("name:", "$name1"), 
                   ("name:", "$name2"), 
                   ("poss:", "$posses")]     

        p = figure(plot_width=500, plot_height=500, 
           sizing_mode="stretch_both", tooltips=tooltip, 
           title=k)
        p.title.text_font_size = '18pt'
        p.circle('x', 'y', size='mark_size', source=source)
        p.xaxis.axis_label = 'Offensive PPP'
        p.yaxis.axis_label = 'Defensive PPP'
        hline = Span(location=season_df.off_wmn.mean(), 
                     dimension="width", line_color='grey', 
                     line_dash='dashed')
        vline = Span(location=season_df.def_wmn.mean(), 
                     dimension="height", line_color='grey', 
                     line_dash='dashed')
        p.renderers.extend([hline, vline])
        save(p)

