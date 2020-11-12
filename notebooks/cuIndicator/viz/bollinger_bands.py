import ipywidgets as widgets
from bqplot.colorschemes import CATEGORY20
from bqplot import Axis, Figure, LinearScale, Lines
import os
from gquant.dataframe_flow.task import load_modules
load_modules(os.getenv('MODULEPATH')+'/rapids_modules/')
from rapids_modules.cuindicator import bollinger_bands as indicator_fun  # noqa #F401


def get_para_widgets():
    para_selector = widgets.IntSlider(
        min=2, max=60, description="Bollinger Bands")
    para_selector_widgets = [para_selector]
    return para_selector_widgets


def get_parameters(stock_df, para_selector_widgets):
    return (stock_df["close"],) + tuple(
        [w.value for w in para_selector_widgets])


def process_outputs(output, stock_df):
    output.b1.index = stock_df.index
    output.b2.index = stock_df.index

    stock_df['out0'] = output.b1
    stock_df['out0'] = stock_df['out0'].fillna(0)
    stock_df['out1'] = output.b2
    stock_df['out1'] = stock_df['out1'].fillna(0)
    return stock_df


def create_figure(stock, dt_scale, sc, color_id,
                  f, indicator_figure_height, figure_width,
                  add_new_indicator):
    sc_co = LinearScale()
    sc_co2 = LinearScale()
    ax_y = Axis(label='Bollinger b1', scale=sc_co, orientation='vertical')
    ax_y2 = Axis(label='Bollinger b2', scale=sc_co2,
                 orientation='vertical', side='right')
    new_line = Lines(x=stock.datetime.to_array(),
                     y=stock['out0'].to_array(),
                     scales={'x': dt_scale, 'y': sc_co},
                     colors=[CATEGORY20[color_id[0]]])
    new_line2 = Lines(x=stock.datetime.to_array(), y=stock['out1'].to_array(),
                      scales={'x': dt_scale, 'y': sc_co2},
                      colors=[CATEGORY20[(color_id[0] + 1) % len(CATEGORY20)]])
    new_fig = Figure(marks=[new_line, new_line2], axes=[ax_y, ax_y2])
    new_fig.layout.height = indicator_figure_height
    new_fig.layout.width = figure_width
    figs = [new_line, new_line2]
    add_new_indicator(new_fig)
    return figs


def update_figure(stock, objects):
    line = objects[0]
    line2 = objects[1]
    with (line.hold_trait_notifications(), line2.hold_trait_notifications()):
        line.y = stock['out0'].to_array()
        line.x = stock.datetime.to_array()
        line2.y = stock['out1'].to_array()
        line2.x = stock.datetime.to_array()
