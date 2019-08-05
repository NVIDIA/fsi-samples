import ipywidgets as widgets


def getXGBoostWidget(replace_spec, run, task_list, outlist, plot_figures):

    def getRangeSlider(val0, val1, des=""):
        return widgets.IntRangeSlider(value=[val0, val1],
                                      min=1,
                                      max=60,
                                      step=1,
                                      description=des,
                                      disabled=False,
                                      continuous_update=False,
                                      orientation='horizontal',
                                      readout=True)

    def getSlider(val, des=""):
        return widgets.IntSlider(value=val,
                                 min=1,
                                 max=60,
                                 step=1,
                                 description=des,
                                 disabled=False,
                                 continuous_update=False,
                                 orientation='horizontal',
                                 readout=True,
                                 readout_format='d')

    out = widgets.Output(layout={'border': '1px solid black'})

    with out:
        indicators = \
            replace_spec['node_technical_indicator']['conf']['indicators']
        chaikin_selector = getRangeSlider(indicators[0]['args'][0],
                                          indicators[0]['args'][1], "Chaikin")

        def chaikin_selection(*stocks):
            with out:
                indicators[0]['args'][0] = chaikin_selector.value[0]
                indicators[0]['args'][1] = chaikin_selector.value[1]
        chaikin_selector.observe(chaikin_selection, 'value')

        bollinger_selector = getSlider(indicators[1]['args'][0], "bollinger")

        def bollinger_selection(*stocks):
            with out:
                indicators[1]['args'][0] = bollinger_selector.value
        bollinger_selector.observe(bollinger_selection, 'value')

        macd_selector = getRangeSlider(indicators[2]['args'][0],
                                       indicators[2]['args'][1],
                                       "MACD")

        def macd_selection(*stocks):
            with out:
                indicators[2]['args'][0] = macd_selector.value[0]
                indicators[2]['args'][1] = macd_selector.value[1]
        macd_selector.observe(macd_selection, 'value')

        rsi_selector = getSlider(indicators[3]['args'][0], "Relative Str")

        def rsi_selection(*stocks):
            with out:
                indicators[3]['args'][0] = rsi_selector.value
        rsi_selector.observe(rsi_selection, 'value')

        atr_selector = getSlider(indicators[4]['args'][0], "ATR")

        def atr_selection(*stocks):
            with out:
                indicators[4]['args'][0] = atr_selector.value
        atr_selector.observe(atr_selection, 'value')

        sod_selector = getSlider(indicators[6]['args'][0], "Sto Osc")

        def sod_selection(*stocks):
            with out:
                indicators[6]['args'][0] = sod_selector.value
        sod_selector.observe(sod_selection, 'value')

        mflow_selector = getSlider(indicators[7]['args'][0], "Money F")

        def mflow_selection(*stocks):
            with out:
                indicators[7]['args'][0] = mflow_selector.value
        mflow_selector.observe(mflow_selection, 'value')

        findex_selector = getSlider(indicators[8]['args'][0], "Force Index")

        def findex_selection(*stocks):
            with out:
                indicators[8]['args'][0] = findex_selector.value
        findex_selector.observe(findex_selection, 'value')

        adis_selector = getSlider(indicators[10]['args'][0], "Ave DMI")

        def adis_selection(*stocks):
            with out:
                indicators[10]['args'][0] = adis_selector.value
        adis_selector.observe(adis_selection, 'value')

        ccindex_selector = getSlider(indicators[11]['args'][0], "Comm Cha")

        def ccindex_selection(*stocks):
            with out:
                indicators[11]['args'][0] = ccindex_selector.value
        ccindex_selector.observe(ccindex_selection, 'value')

        bvol_selector = getSlider(indicators[12]['args'][0], "On Balance")

        def bvol_selection(*stocks):
            with out:
                indicators[12]['args'][0] = bvol_selector.value
        bvol_selector.observe(bvol_selection, 'value')

        vindex_selector = getSlider(indicators[13]['args'][0], "Vortex")

        def vindex_selection(*stocks):
            with out:
                indicators[13]['args'][0] = vindex_selector.value
        vindex_selector.observe(vindex_selection, 'value')

        mindex_selector = getRangeSlider(indicators[15]['args'][0],
                                         indicators[15]['args'][1],
                                         "Mass Index")

        def mindex_selection(*stocks):
            with out:
                indicators[15]['args'][0] = mindex_selector.value[0]
                indicators[15]['args'][1] = mindex_selector.value[1]
        mindex_selector.observe(mindex_selection, 'value')

        tindex_selector = getRangeSlider(indicators[16]['args'][0],
                                         indicators[16]['args'][1],
                                         "True Strength")

        def tindex_selection(*stocks):
            with out:
                indicators[16]['args'][0] = tindex_selector.value[0]
                indicators[16]['args'][1] = tindex_selector.value[1]
        tindex_selector.observe(tindex_selection, 'value')

        emove_selector = getSlider(indicators[17]['args'][0], "Easy Move")

        def emove_selection(*stocks):
            with out:
                indicators[17]['args'][0] = emove_selector.value
        emove_selector.observe(emove_selection, 'value')

        cc_selector = getSlider(indicators[18]['args'][0], "Cppock Curve")

        def cc_selection(*stocks):
            with out:
                indicators[18]['args'][0] = cc_selector.value
        cc_selector.observe(cc_selection, 'value')

        kchannel_selector = getSlider(indicators[19]['args'][0],
                                      "Keltner Channel")

        def kchannel_selection(*stocks):
            with out:
                indicators[19]['args'][0] = kchannel_selector.value
        kchannel_selector.observe(kchannel_selection, 'value')

        button = widgets.Button(
                                description='Compute',
                                disabled=False,
                                button_style='',
                                tooltip='Click me')

        def on_button_clicked(b):
            with out:
                print("Button clicked.")
                w.children = (w.children[0], widgets.Label("Busy...."),)
                o_gpu = run(task_list, outputs=outlist,
                            replace=replace_spec)
                figure_combo = plot_figures(o_gpu)
                w.children = (w.children[0], figure_combo,)
        button.on_click(on_button_clicked)

    selectors = widgets.VBox([chaikin_selector, bollinger_selector,
                              macd_selector, rsi_selector, atr_selector,
                              sod_selector, mflow_selector, findex_selector,
                              adis_selector, ccindex_selector, bvol_selector,
                              vindex_selector, mindex_selector,
                              tindex_selector, emove_selector, cc_selector,
                              kchannel_selector, button])
    w = widgets.VBox([selectors])
    return w
