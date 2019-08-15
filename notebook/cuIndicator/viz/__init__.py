import ipywidgets as widgets

def get_para_widgets():
    para_selector = widgets.IntSlider(min=2, max=60, description="TRIX")
    para_selector_widgets = [para_selector]                    
    return para_selector_widgets

def get_parameters(stock_df, para_selector_widgets):
    return  (stock_df["close"],) + tuple([w.value for w in para_selector_widgets])

def process_outputs(output, stock_df):
    stock_df['out'] = output
    stock_df['out'] = stock_df['out'].fillna(0)
    return stock_df