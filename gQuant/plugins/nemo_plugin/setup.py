from setuptools import setup, find_packages

setup(
    name='gquant_nemo_plugin',
    packages=find_packages(include=['gquant_nemo_plugin',
                                    'gquant_nemo_plugin.nemo_util']),
    entry_points={
        'gquant.plugin':
        ['gquant_nemo_plugin = gquant_nemo_plugin',
         'gquant_nemo_plugin.asr = gquant_nemo_plugin.asr',
         'gquant_nemo_plugin.cv = gquant_nemo_plugin.cv',
         'gquant_nemo_plugin.nlp = gquant_nemo_plugin.nlp',
         'gquant_nemo_plugin.util = gquant_nemo_plugin.nemo_util',
         'gquant_nemo_plugin.gan = gquant_nemo_plugin.simple_gan',
         'gquant_nemo_plugin.tts = gquant_nemo_plugin.tts',
         'gquant_nemo_plugin.tutorials = gquant_nemo_plugin.tutorials'],
    }
)
