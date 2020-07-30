# gQuant - GPU Accelerated Graph Computation for Quantitative Analyst Tasks

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/gquant/blob/master/README.md) ensure you are on the `master` branch.

## What is gQuant?
gQuant is a collection of open-source GPU accelerated Python tools and examples  for quantitative analyst tasks,  built on top of the [RAPIDS AI](https://rapids.ai/) project, [Numba](https://numba.pydata.org/), and [Dask](https://dask.org/).

The examples range from simple accelerated calculation of technical trading indicators through defining workflows for interactively developing trading strategies and automating many typical tasks.

The extensibility of the system is highlighted by examples showing how to create a dataframe flow graph, which allows for easy re-use and composability of higher level workflows.

The examples also show how to easily convert a single-threaded solution into a Dask distributed one. 

These examples can be used as-is or, as they are open source, can be extended to suit your environments.

## gQuant jupyterlab extension
The gQuant juyterlab extension provides the user interface to build the dataframe flow taskgraph easily. It takes advantage of the open sources projects like [jupyterlab](https://github.com/jupyterlab/jupyterlab), [ipywidget](https://github.com/jupyter-widgets/ipywidgets), [React](https://reactjs.org/) and [D3](https://d3js.org/). It features:
1. Takes full advantage of JupyterLab project that the extension provides context menu, command palette commands, keyboard shortcuts to speed up the productivity.  
2. Define a new TaskGraph file format `.gq.yaml` that can be edited in the Jupyterlab. 
3. Visually presents the TaskGraph as a DAG graph. User can zoom in and out, freely move the nodes around and make connections between nodes.
4. Use the special `Ouput Collector` to gather the results and organize them in a tab widget. The IPython [rich display](https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display) is fully supported.
5. Visually shows the progress of evaluating the graph. The node computation dependence is clearly shown.
6. Automatically generate the UI elements to edit the Node configuration given the configuration JSON schema. 
7. Dynamically compute the input output ports compatability, dataframe columns names and types, ports types to prevent connection errors. 
8. Nodes can have multiple output ports that can be used to generate different output types. E.g. it can provides both `cudf` and `dask_cudf` output ports so the nodes connected to `dask_cudf` port are doing distributed computation automatically. 
9. Provides the standard API to extend your own computation Nodes.
![Quick Demo](gquantlab_demo.gif "Demo")


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

### Install

gQuant source code can be downloaded from [GitHub](https://github.com/rapidsai/gquant).

- Git clone source code:

```bash
$ git clone https://github.com/rapidsai/gQuant.git
```


- Build and run the container:
```bash
$ cd gQuant/docker && . build.sh
$ docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 gquant/gquant:[tag from the build]
$ bash rapids/notebooks/utils/start-jupyter.sh 
```

### Example notebooks

Example notebooks, tutorial showcasing, can be found in __notebooks__ folder.
