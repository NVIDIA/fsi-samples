## Greenflow Dask Plugin 

This is a Greenflow plugin that includes a set of utlility nodes for Dask workflows. 

Greenflow take advantage of the `entry point` inside the `setup.py` file to register the plugin. Greenflow can discover all the plugins that has the entry point group name `greenflow.plugin`. Check the `setup.py` file to see details.

### Create an new Python enviroment
```bash
conda create -n test python=3.8
```

### Install the greenflowlab JupyterLab plugin
To install `greenflowlab` JupyterLab plugin, make sure `nodejs` of version [12^14^15] is installed. E.g:
```bash
conda install -c conda-forge python-graphviz nodejs=12.4.0 pydot
```
Then install the `greenflowlab`:
```bash
pip install greenflowlab
```
Or install `greenflowlab` at the greenflowlab directory:
```bash
pip install .
```

### Install the external example plugin
To install the external plugin, in the plugin diretory, run following command
```bash
pip install .
```

