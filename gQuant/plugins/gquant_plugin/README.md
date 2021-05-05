## Greenflow RAPIDS Plugin Example
The examples range from simple accelerated calculation of technical trading indicators through defining workflows for interactively developing trading strategies and automating many typical tasks.

The extensibility of the system is highlighted by examples showing how to create a taskgraph workflow, which allows for easy re-use and composability of higher level workflows.

The examples also show how to easily convert a single-threaded solution into a Dask distributed one. 

These examples can be used as-is or, as they are open source, can be extended to suit your environments.

Greenflow take advantage of the `entry point` inside the `setup.py` file to register the plugin. Greenflow can discover all the plugins that has the entry point group name `greenflow.plugin`. Check the `setup.py` file to see details.

### Create an new Python enviroment
```bash
conda create -n test python=3.8
```

### Prerequisites
- NVIDIA Pascal™ GPU architecture or better.
- [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive) with driver v396.37+ or [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) with driver v410.48+.
- Ubuntu 16.04 or 18.04.
- [NVIDIA-docker v2+](https://github.com/nvidia/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-20-if-im-not-using-the-latest-docker-version).


### Download data files

Run the following command at the project root diretory 
```bash
bash download_data.sh

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
Install RAPIDS:
```bash
conda install -y -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.19
```
To install the external plugin, in the plugin diretory, run following command
```bash
pip install .
```

### Launch the Jupyter lab
After launching the JupyterLab by,
```bash
MODULEPATH=modules jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```
