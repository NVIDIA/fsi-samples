import ipywidgets as widgets
from bqplot.colorschemes import CATEGORY20
from bqplot import Axis, Figure, LinearScale, Lines

from gquant.cuindicator import kst_oscillator as indicator_fun

def get_para_widgets():
    para_selector = widgets.IntSlider(min=2, max=60, description="KST Oscillator")
    para_selector_widgets = [para_selector]
    return para_selector_widgets

def get_parameters(stock_df, para_selector_widgets):
    param = [w.value for w in para_selector_widgets]
    print (param)
    param_grp = [param[0] + i for i in range(8)]
    print (param_grp)
    return (stock_df["close"],param_grp[0],param_grp[1],param_grp[2],param_grp[3],
            param_grp[4],param_grp[5],param_grp[6],param_grp[7])

def process_outputs(output,stock_df):
    stock_df['out'] = output
    stock_df['out'] = output.fillna(0)
    return stock_df

def create_figure(stock, dt_scale, sc, color_id, f, indicator_figure_height, figure_width, add_new_indicator):
    sc_co = LinearScale()
    ax_y = Axis(label='KST Oscillator',scale=sc_co, orientation='vertical')
    new_line = Lines(x=stock.datetime, y=stock['out'], scales={'x': dt_scale, 'y': sc_co}, colors=[CATEGORY20[color_id[0]]])
    new_fig = Figure(marks=[new_line], axes=[ax_y])
    new_fig.layout.height = indicator_figure_height
    new_fig.layout.width = figure_width                    
    figs = [new_line]
    # add new figure
    add_new_indicator(new_fig)
    return figs
