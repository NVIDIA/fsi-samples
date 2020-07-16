# gQuant - GPU Accelerated Framework for Quantitative Analyst Tasks

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/gquant/blob/main/README.md) ensure you are on the `main` branch.

## What is gQuant?
gQuant is a collection of open-source GPU accelerated Python tools and examples  for quantitative analyst tasks,  built on top of the [RAPIDS AI](https://rapids.ai/) project, [Numba](https://numba.pydata.org/), and [Dask](https://dask.org/).

The examples range from simple accelerated calculation of technical trading indicators through defining workflows for interactively developing trading strategies and automating many typical tasks.

The extensibility of the system is highlighted by examples showing how to create a dataframe flow graph, which allows for easy re-use and composability of higher level workflows.

The examples also show how to easily convert a single-threaded solution into a Dask distributed one. 

These examples can be used as-is or, as they are open source, can be extended to suit your environments.

---
## Getting started

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
$ docker run --runtime=nvidia --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 gquant/gquant:latest
$ bash rapids/notebooks/utils/start-jupyter.sh 
```

### Example notebooks

Example notebooks, tutorial showcasing, can be found in __notebooks__ folder.
