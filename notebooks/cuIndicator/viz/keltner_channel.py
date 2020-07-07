import ipywidgets as widgets
from bqplot import Axis, Figure, LinearScale, Lines
from gquant.cuindicator import keltner_channel as indicator_fun  # noqa #F401


def get_para_widgets():
    para_selector = widgets.IntSlider(min=2, max=60,
                                      description="Keltner Channel")
    para_selector_widgets = [para_selector]
    return para_selector_widgets


def get_parameters(stock_df, para_selector_widgets):
    return (stock_df["high"],
            stock_df["low"],
            stock_df["close"]) + tuple([
                w.value for w in para_selector_widgets])


def process_outputs(output, stock_df):
    output.KelChM.index = stock_df.index
    output.KelChU.index = stock_df.index
    output.KelChD.index = stock_df.index
    stock_df['out0'] = output.KelChM
    stock_df['out0'] = stock_df['out0'].fillna(0)
    stock_df['out1'] = output.KelChU
    stock_df['out1'] = stock_df['out1'].fillna(0)
    stock_df['out2'] = output.KelChD
    stock_df['out2'] = stock_df['out2'].fillna(0)
    return stock_df


def create_figure(stock, dt_scale,
                  sc, color_id, f,
                  indicator_figure_height, figure_width, add_new_indicator):
    sc_co = LinearScale()
    ax_y = Axis(label='Keltner Channel', scale=sc_co, orientation='vertical')
    new_line = Lines(x=stock.datetime.to_array(), y=[stock['out0'].to_array(),
                                                     stock['out1'].to_array(),
                                                     stock['out2'].to_array()],
                     scales={'x': dt_scale, 'y': sc_co})
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
        line.y = [stock['out0'].to_array(), stock['out1'].to_array(),
                  stock['out2'].to_array()]
        line.x = stock.datetime.to_array()
