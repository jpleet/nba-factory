import numpy as np
import pandas as pd 
from bokeh.plotting import ColumnDataSource, figure, output_file, save
from bokeh.models import Span

import sys
sys.path.append('../')
from params import *

def plot(df):

    # assume season in df
    if verbose:
        print('Making individual plots')

    for k, pred_df in df.groupby('seasons'):

        # format season parameter
        if len(k) == 0:
            s = min(pbp_seasons) - 1
            e = max(pbp_seasons)
            k = f'{s}-{e}'
        elif len(k) == 1:
            k = k[0][0:4] + '-' + k[0][4:]
        else:
            k = ', '.join(i[:4]+'-'+i[4:] for i in k)
        # scale marker sizes
        marker_size = (((pred_df.total_poss - 
                        pred_df.total_poss.min()) / 
                       (pred_df.total_poss.max() - 
                        pred_df.total_poss.min())) + 0.1) * 10
        # set up player images
        base = ('https://ak-static.cms.nba.com/wp-content/' + 
                'uploads/headshots/nba/latest/260x190')
        imgs = [f"{base}/{pid}.png" for pid in pred_df.player_id]
        # create plot
        sfile = f'{prediction_folder}/player_ppp_{k}.html'
        output_file(sfile)

        source = ColumnDataSource(data=dict(
            x = pred_df.off_wmn.values, 
            y = pred_df.def_wmn.values, 
            names = pred_df.name.values, 
            posses = pred_df.total_poss.values, 
            imgs = imgs, 
            mark_size = marker_size))

        tooltip = """
            <div>
                <div>
                    <img
                        src="@imgs" height="95" alt="@imgs" width="130"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-weight: bold;">@names</span>
                </div>
                <div>
                    <span>(@posses)</span>
                </div>
            </div>
        """

        p = figure(plot_width=500, plot_height=500, 
                   sizing_mode="stretch_both", tooltips=tooltip, 
                   title=k)
        p.title.text_font_size = '18pt'
        p.circle('x', 'y', size='mark_size', source=source)
        p.xaxis.axis_label = 'Offensive PPP'
        p.yaxis.axis_label = 'Defensive PPP'
        hline = Span(location=pred_df.off_wmn.mean(), 
                     dimension="width", line_color='grey', 
                     line_dash='dashed')
        vline = Span(location=pred_df.def_wmn.mean(), 
                     dimension="height", line_color='grey', 
                     line_dash='dashed')
        p.renderers.extend([hline, vline])
        save(p)
