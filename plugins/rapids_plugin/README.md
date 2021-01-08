## gQuant RAPIDS Plugin Example

This is a example to show how to write an external gQuant RAPIDS plugin. gQuant take advantage of the `entry point` inside the `setup.py` file to register the plugin. gQuant can discover all the plugins that has the entry point group name `gquant.plugin`. Check the `setup.py` file to see details.

### Create an new Python enviroment
```bash
conda create -n test python=3.8
```

### Install the gQuant 
To install the gQuant graph computation library, first install the dependence libraries:
```bash
conda install -c conda-forge dask networkx python-graphviz ruamel.yaml pandas pydot
```

Then install `gquant`:
```bash
pip install gquant
```
Or install `gquant` at the root directory:
```bash
pip install .
```

### Install the gquantlab JupyterLab plugin
To install `gquantlab` JupyterLab plugin, install the following dependence libraries:
```bash
conda install -c conda-forge ipywidgets nodejs=12.4.0
```
Then install the `gquantlab`:
```bash
pip install gquantlab
```
Or install `gquantlab` at the gquantlab directory:
```bash
pip install .
```

### Install the external example plugin
```
conda install -y -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.17
conda install -c conda-forge yarn pytables
pip install git+https://github.com/bqplot/bqplot.git@master
pip install ray[tune]
```
To install the external plugin, in the plugin diretory, run following command
```bash
pip install .
```

### Launch the Jupyter lab
After launching the JupyterLab by,
```bash
jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```
