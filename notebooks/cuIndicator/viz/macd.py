import ipywidgets as widgets
from bqplot.colorschemes import CATEGORY20
from bqplot import Axis, Figure, LinearScale, Lines

from gquant.cuindicator import macd as indicator_fun

def get_para_widgets():
    para_selector = widgets.IntRangeSlider(value=[10, 30],
                                            min=3,
                                            max=60,
                                            step=1,
                                            description="MACD:",
                                            disabled=False,
                                            continuous_update=False,
                                            orientation='horizontal',
                                            readout=True)
    para_selector_widgets = [para_selector]                    
    return para_selector_widgets

def get_parameters(stock_df, para_selector_widgets):
    widget = para_selector_widgets[0]
    return  (stock_df["close"], widget.value[0], widget.value[1])

def process_outputs(output, stock_df):
    stock_df['out0'] = output.MACD
    stock_df['out0'] = stock_df['out0'].fillna(0)
    stock_df['out1'] = output.MACDsign
    stock_df['out1'] = stock_df['out1'].fillna(0)
    stock_df['out2'] = output.MACDdiff
    stock_df['out2'] = stock_df['out2'].fillna(0)
    return stock_df

def create_figure(stock, dt_scale, sc, color_id, f, indicator_figure_height, figure_width, add_new_indicator):
    sc_co = LinearScale()
    ax_y = Axis(label='MACD', scale=sc_co, orientation='vertical')
    new_line = Lines(x=stock.datetime, y=[stock['out0'], stock['out1'], stock['out2'] ], scales={'x': dt_scale, 'y': sc_co}) #
    new_fig = Figure(marks=[new_line], axes=[ax_y])
    new_fig.layout.height = indicator_figure_height
    new_fig.layout.width = figure_width                    
    figs = [new_line]
    # add new figure
    add_new_indicator(new_fig)                    
    return figs

def update_figure(stock, objects):
    line = objects[0]
    with line.hold_trait_notifications():
        line.y = [stock['out0'], stock['out1'], stock['out2']]
        line.x = stock.datetime