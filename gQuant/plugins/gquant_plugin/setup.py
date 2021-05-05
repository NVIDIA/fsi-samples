from setuptools import setup, find_packages

setup(
    name='greenflow_gquant_plugin',
    version='0.0.2',
    install_requires=[
        "greenflow", "bqplot", "tables", "ray[tune]", "matplotlib", "ray[default]",
        "mplfinance"
    ],
    packages=find_packages(include=[
        'greenflow_gquant_plugin', 'greenflow_gquant_plugin.analysis',
        'greenflow_gquant_plugin.backtest',
        'greenflow_gquant_plugin.dataloader', 'greenflow_gquant_plugin.ml',
        'greenflow_gquant_plugin.portofolio',
        'greenflow_gquant_plugin.strategy',
        'greenflow_gquant_plugin.cuindicator',
        'greenflow_gquant_plugin.transform'
    ]),
    entry_points={
        'greenflow.plugin': [
            'greenflow_gquant_plugin = greenflow_gquant_plugin',
            'greenflow_gquant_plugin.analysis = greenflow_gquant_plugin.analysis',
            'greenflow_gquant_plugin.backtest = greenflow_gquant_plugin.backtest',
            'greenflow_gquant_plugin.dataloader = greenflow_gquant_plugin.dataloader',
            'greenflow_gquant_plugin.ml = greenflow_gquant_plugin.ml',
            'greenflow_gquant_plugin.portofolio = greenflow_gquant_plugin.portofolio',
            'greenflow_gquant_plugin.strategy = greenflow_gquant_plugin.strategy',
            'greenflow_gquant_plugin.transform = greenflow_gquant_plugin.transform'
        ],
    })
