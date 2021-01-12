# gQuant - Graph Computation Tool

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/gquant/blob/main/README.md) ensure you are on the `main` branch.


## What's Inside This Repo

There are a few projects inside this repo:

1. `gquant` -  A graph computation toolkit that helps you to organize the workflows in graph computation.
2. `gquantlab` - A JupyterLab plugin that provides the UI interface for `gquant`.
3. `plugins` - A few gquant plugins with example notebooks. 
  1. `simple_example` - A simple external plugin example for gQuant.
  2. `rapids_plugin` - An external plugin with a set of nodes for quantitative analyst tasks, built on top of the [RAPIDS AI](https://rapids.ai/) project, [Numba](https://numba.pydata.org/), and [Dask](https://dask.org/).
  3. `nemo_plugin` - An external plugin with a set of nodes that wraps the [NeMo library](https://github.com/NVIDIA/NeMo) . 

These projects are all released as independent Python projects with their own `setup.py` files. 

## Screenshots
![Tuturial](tutorial.gif "Tutorial")
![Quick Demo](gquantlab_demo.gif "Demo")


## Binary installation

### Install the gquant JupyterLab plugin
To install the gQuant graph computation library, install the dependence libraries:
```bash
conda install dask networkx python-graphviz ruamel.yaml pandas
```
Then install `gquant`:
```bash
pip install gquant
```
Or install `gquant` at the gquant directory:
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

### Install the gquant plugins

Under the plugin root directory, run following command to install them:
```bash
pip install .
```

Note, gQuant node plugins can be registered in two ways: 

  1. Register the plugin in `gquantrc` file. Check the `System environment` for details
  2. Write a external plugin using 'entry point' to register it. Check the `plugins` directory for details


## Docker Install

- Build and run the container:

```bash
$ cd gQuant/docker && . build.sh
```
When building the container, you can run gQuant in two modes: dev or prod. In the dev mode, please check the README file in `gquantlab` directory to install the plugins and Python libraries. 

In the production mode, you can launch the container by following command and start to use it 
```bash
$ docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 gquant/gquant:[tag from the build]
```

## Example notebooks

Example notebooks, tutorial showcasing, can be found in __notebooks__ folder in the plugin directory.


## System environment 

There are a few system environment that the user can overwrite. 

The custom module files are specified in the `gquantrc` file. `GQUANT_CONFIG` enviroment variable points to the location of this file. By default, it points to 
`$CWD\gquantrc`. 

In the example `gquantrc`, system environment variable `MODULEPATH` is used to point to the paths of the module files.
To start the jupyterlab, please make sure `MODULEPATH` is set properly. 

For example, if you want to start the jupyterlab in the gQuant root directory.
```bash
MODULEPATH=$PWD/modules jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

Or, if you want to start the jupyterlab in the gquantlab directory.
```bash
GQUANT_CONFIG=../gquantrc MODULEPATH=$PWD/../modules jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```
