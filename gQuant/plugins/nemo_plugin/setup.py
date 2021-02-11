from setuptools import setup, find_packages

setup(
    name='greenflow_nemo_plugin',
    packages=find_packages(include=['greenflow_nemo_plugin',
                                    'greenflow_nemo_plugin.nemo_util']),
    entry_points={
        'greenflow.plugin':
        ['greenflow_nemo_plugin = greenflow_nemo_plugin',
         'greenflow_nemo_plugin.asr = greenflow_nemo_plugin.asr',
         'greenflow_nemo_plugin.cv = greenflow_nemo_plugin.cv',
         'greenflow_nemo_plugin.nlp = greenflow_nemo_plugin.nlp',
         'greenflow_nemo_plugin.util = greenflow_nemo_plugin.nemo_util',
         'greenflow_nemo_plugin.gan = greenflow_nemo_plugin.simple_gan',
         'greenflow_nemo_plugin.tts = greenflow_nemo_plugin.tts',
         'greenflow_nemo_plugin.tutorials = greenflow_nemo_plugin.tutorials'],
    }
)
