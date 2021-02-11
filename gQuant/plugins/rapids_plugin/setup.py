from setuptools import setup, find_packages

setup(
    name='gquant_rapids_plugin',
    install_requires=[
        "bqplot", "tables", "ray[tune]"
    ],
    packages=find_packages(include=['gquant_rapids_plugin',
                                    'gquant_rapids_plugin.analysis',
                                    'gquant_rapids_plugin.backtest',
                                    'gquant_rapids_plugin.dataloader',
                                    'gquant_rapids_plugin.ml',
                                    'gquant_rapids_plugin.portofolio',
                                    'gquant_rapids_plugin.strategy',
                                    'gquant_rapids_plugin.cuindicator',
                                    'gquant_rapids_plugin.transform']),
    entry_points={
        'gquant.plugin':
        ['gquant_rapids_plugin = gquant_rapids_plugin',
         'gquant_rapids_plugin.analysis = gquant_rapids_plugin.analysis',
         'gquant_rapids_plugin.backtest = gquant_rapids_plugin.backtest',
         'gquant_rapids_plugin.dataloader = gquant_rapids_plugin.dataloader',
         'gquant_rapids_plugin.ml = gquant_rapids_plugin.ml',
         'gquant_rapids_plugin.portofolio = gquant_rapids_plugin.portofolio',
         'gquant_rapids_plugin.strategy = gquant_rapids_plugin.strategy',
         'gquant_rapids_plugin.transform = gquant_rapids_plugin.transform'],
    }
)
