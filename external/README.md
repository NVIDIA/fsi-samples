## Simple External Plugin Example

This is a simple example to show how to write an external gQuant plugin. gQuant take advantage of the `entry point` inside the `setup.py` file to register the plugin. gQuant can discover all the plugins that has the entry point group name `gquant.plugin`. Check the `setup.py` file to see details.

### Create an new Python enviroment
```bash
conda create -n test python=3.8
```

### Install the gQuant lib
To install the gQuant graph computation library, first install the dependence libraries:
```bash
pip install dask[dataframe] distributed networkx
conda install python-graphviz ruamel.yaml numpy pandas
```
Then install gquant lib:
```bash
pip install gquant
```

### Install the gQuantlab plugin
To install JupyterLab plugin, install the following dependence libraries:
```bash
conda install nodejs ipywidgets
```
Then install the gquantlab lib:
```bash
pip install gquantlab==0.1.2
```
Build the ipywidgets Jupyterlab plugin
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0
```
If you launch the JupyterLab, it will prompt to build the new plugin. You can
explicitly build it by:
```bash
jupyter lab build
```

### Install the external example plugin
To install the external plugin, in the plugin diretory, run following command
```bash
pip install .
```

### Launch the Jupyter lab
After launching the JupyterLab by,
```bash
jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```
You can see the `DistanceNode` and `PointNode` under the name `custom_node` in the menu.
