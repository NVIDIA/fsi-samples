# gquantlab

![Github Actions Status](https://github.com/rapidsai/gQuant/gquantlab/workflows/Build/badge.svg)

## gQuant jupyterlab extension
The gQuant Juyterlab extension provides the user interface to build the dataframe flow TaskGraph easily. It takes advantage of the open sources projects like [jupyterlab](https://github.com/jupyterlab/jupyterlab), [ipywidget](https://github.com/jupyter-widgets/ipywidgets), [React](https://reactjs.org/) and [D3](https://d3js.org/). It features:
1. Takes full advantage of the JupyterLab project that the extension adds commands to Jupyterlab context menu, command palette and bind them with keyboard shortcuts to speed up the productivity.  
2. Define a new TaskGraph file format `.gq.yaml` that can be edited in the Jupyterlab. 
3. Visually presents the TaskGraph as a DAG graph. Users can zoom in and out, freely move the nodes around, and make connections between nodes.
4. Use the special `Ouput Collector` to gather the results and organize them in a tab widget. The IPython [rich display](https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display) is fully supported.
5. Visually shows the progress of graph evaluation and computation dependence.
6. Automatically generate the UI elements to edit and validate the Node configuration given the configuration JSON schema. It exposes the function API in a user-friendly way. User can change the configuration and re-run the computation to test out the hyperparameters easily.
7. Dynamically compute the input-output ports compatibility, dataframe columns names and types, ports types to prevent connection errors. 
8. Nodes can have multiple output ports that can be used to generate different output types. E.g. some data loader Node provides both `cudf` and `dask_cudf` output ports. The multiple GPUs distributed computation computation is automatically enabled by switching to the `dask_cudf` output port. 
9. Provides the standard API to extend your computation Nodes.
10. The composite node can encapsulate the TaskGraph into a single node for easy reuse. The composite node can be exported as a regular gQuant node without any coding.


This extension is composed of a Python package named `gquantlab`
for the server extension and a NPM package named `gquantlab`
for the frontend extension.


## Build the dev container

In the gQuant root direction
```bash
bash docker/build.sh
```
Launch your development container, make sure mounting your gQuant directory
to the container and open `8888` ports.

Set the gQuant path as the folder to start the development or you can open the 
`workspace.code-workspace` file.


## Requirements

* JupyterLab >= 2.0

## Install

Note: You will need NodeJS of version 12^14^15 to install the extension.

```bash
pip install gquantlab
```

## Troubleshoot

If you are seeing the frontend extension but it is not working, check
that the server extension is enabled:

```bash
jupyter serverextension list
```

If the server extension is installed and enabled but you are not seeing
the frontend, check the frontend is installed:

```bash
jupyter labextension list
```

If it is installed, try:

```bash
jupyter lab clean
jupyter lab build
```

## Contributing

### Install

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Move to gquantlab directory

# Install server extension
pip install -e .
# Register server extension
jupyter serverextension enable --py gquantlab --sys-prefix

# Install dependencies
jlpm
# Build Typescript source
jlpm build
# Link your development version of the extension with JupyterLab
jupyter labextension install .
# Rebuild Typescript source after making changes
jlpm build
# Rebuild JupyterLab after making any changes
jupyter lab build
```

You can watch the source directory and run JupyterLab in watch mode to watch for changes in the extension's source and automatically rebuild the extension and application.

```bash
# Watch the source directory in another terminal tab
jlpm watch
# Run jupyterlab in watch mode in one terminal tab
jupyter lab --watch
```

Now every change will be built locally and bundled into JupyterLab. Be sure to refresh your browser page after saving file changes to reload the extension (note: you'll need to wait for webpack to finish, which can take 10s+ at times).

### Uninstall

```bash
pip uninstall gquantlab
jupyter labextension uninstall gquantlab
```
