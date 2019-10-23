import ipywidgets as widgets
from bqplot.colorschemes import CATEGORY20
from bqplot import Lines
import math

from gquant.cuindicator import moving_average as indicator_fun

def get_para_widgets():
    para_selector = widgets.IntSlider(min=2, max=60, description="avg periods")
    para_selector_widgets = [para_selector]
    
    return para_selector_widgets

def get_parameters(stock_df, para_selector_widgets):
    return (stock_df["close"],) + tuple([w.value for w in para_selector_widgets])

def process_outputs(output, stock_df):
    stock_df['out'] = output
    stock_df['out'] = stock_df['out'].fillna(math.inf)
    return stock_df

def create_figure(stock, dt_scale, sc, color_id, f, indicator_figure_height, figure_width, add_new_indicator):
    line = Lines(x=stock.datetime, y=stock['out'], scales={'x': dt_scale, 'y': sc}, colors=[CATEGORY20[color_id[0]]])
    figs = [line]
    f.marks = f.marks + figs
    return figs
