import ipywidgets as widgets
from bqplot.colorschemes import CATEGORY20
from bqplot import Axis, Figure, LinearScale, Lines

from gquant.cuindicator import ppsr as indicator_fun

def get_para_widgets():
    #para_selector = widgets.IntSlider(min=2, max=60, description="Parabolic SAR")
    para_selector_widgets = []
    return para_selector_widgets

def get_parameters(stock_df,para_selector_widgets):
    return (stock_df["high"], stock_df["low"], stock_df["close"]) 

def process_outputs(output,stock_df):
    stock_df['out0'] = output.PP
    stock_df['out0'] = stock_df['out0'].fillna(0)
    stock_df['out1'] = output.R1
    stock_df['out1'] = stock_df['out1'].fillna(0)
    stock_df['out2'] = output.S1
    stock_df['out2'] = stock_df['out2'].fillna(0)
    stock_df['out3'] = output.R2
    stock_df['out3'] = stock_df['out3'].fillna(0)
    stock_df['out4'] = output.S2
    stock_df['out4'] = stock_df['out4'].fillna(0)
    stock_df['out5'] = output.R3
    stock_df['out5'] = stock_df['out5'].fillna(0)
    stock_df['out6'] = output.S3
    stock_df['out6'] = stock_df['out6'].fillna(0)
    return stock_df

def create_figure(stock, dt_scale, sc, color_id, f, indicator_figure_height, figure_width, add_new_indicator):
    sc_co = LinearScale()
    sc_co2 = LinearScale()
    sc_co3 = LinearScale()
    sc_co4 = LinearScale()
    sc_co5 = LinearScale()
    sc_co6 = LinearScale()
    sc_co7 = LinearScale()
    
    ax_y = Axis(label='PPSR PP', scale=sc_co, orientation='vertical')
    ax_y2 = Axis(label='PPSR R1', scale=sc_co2, orientation='vertical', side='right')
    ax_y3 = Axis(label='PPSR S1', scale=sc_co3, orientation='vertical', side='right')
    ax_y4 = Axis(label='PPSR R2', scale=sc_co4, orientation='vertical', side='right')
    ax_y5 = Axis(label='PPSR S2', scale=sc_co5, orientation='vertical', side='right')
    ax_y6 = Axis(label='PPSR R3', scale=sc_co6, orientation='vertical', side='right')
    ax_y7 = Axis(label='PPSR S3', scale=sc_co7, orientation='vertical', side='right')
    new_line = Lines(x=stock.datetime, y=stock['out0'], scales={'x': dt_scale, 'y': sc_co}, 
                        colors=[CATEGORY20[color_id[0]]]) 
    new_line2 = Lines(x=stock.datetime, y=stock['out1'], scales={'x': dt_scale, 'y': sc_co2}, 
                        colors=[CATEGORY20[(color_id[0] + 1) % len(CATEGORY20)]]) 
    new_line3 = Lines(x=stock.datetime, y=stock['out2'], scales={'x': dt_scale, 'y': sc_co3}, 
                        colors=[CATEGORY20[(color_id[0] + 2) % len(CATEGORY20)]])
    new_line4 = Lines(x=stock.datetime, y=stock['out3'], scales={'x': dt_scale, 'y': sc_co4}, 
                        colors=[CATEGORY20[(color_id[0] + 3) % len(CATEGORY20)]]) 
    new_line5 = Lines(x=stock.datetime, y=stock['out4'], scales={'x': dt_scale, 'y': sc_co5}, 
                        colors=[CATEGORY20[(color_id[0] + 4) % len(CATEGORY20)]]) 
    new_line6 = Lines(x=stock.datetime, y=stock['out5'], scales={'x': dt_scale, 'y': sc_co6}, 
                        colors=[CATEGORY20[(color_id[0] + 5) % len(CATEGORY20)]]) 
    new_line7 = Lines(x=stock.datetime, y=stock['out6'], scales={'x': dt_scale, 'y': sc_co7}, 
                        colors=[CATEGORY20[(color_id[0] + 6) % len(CATEGORY20)]]) 

    
    new_fig = Figure(marks=[new_line, new_line2, new_line3, new_line4, 
                            new_line5, new_line6, new_line7], 
                        axes=[ax_y, ax_y2, ax_y3, ax_y4, ax_y5, ax_y6, ax_y7])
    new_fig.layout.height = indicator_figure_height
    new_fig.layout.width = figure_width                    
    figs = [new_line, new_line2, new_line3, new_line4, new_line5, new_line6, new_line7]
    add_new_indicator(new_fig)
    return figs

def update_figure(stock, objects):
    line = objects[0]
    line2 = objects[1]
    with line.hold_trait_notifications() as lc, line2.hold_trait_notifications() as lc2:
        line.y = stock['out0']
        line.x = stock.datetime        
        line2.y = stock['out1']
        line2.x = stock.datetime      
