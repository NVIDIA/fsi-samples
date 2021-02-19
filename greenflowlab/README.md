# greenflowlab

![Github Actions Status](https://github.com/rapidsai/greenflow/greenflowlab/workflows/Build/badge.svg)

## greenflow jupyterlab extension
The greenflow Juyterlab extension provides the user interface to build the dataframe flow TaskGraph easily. It takes advantage of the open sources projects like [jupyterlab](https://github.com/jupyterlab/jupyterlab), [ipywidget](https://github.com/jupyter-widgets/ipywidgets), [React](https://reactjs.org/) and [D3](https://d3js.org/). It features:
1. Takes full advantage of the JupyterLab project that the extension adds commands to Jupyterlab context menu, command palette and bind them with keyboard shortcuts to speed up the productivity.  
2. Define a new TaskGraph file format `.gq.yaml` that can be edited in the Jupyterlab. 
3. Visually presents the TaskGraph as a DAG graph. Users can zoom in and out, freely move the nodes around, and make connections between nodes.
4. Use the special `Ouput Collector` to gather the results and organize them in a tab widget. The IPython [rich display](https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display) is fully supported.
5. Visually shows the progress of graph evaluation and computation dependence.
6. Automatically generate the UI elements to edit and validate the Node configuration given the configuration JSON schema. It exposes the function API in a user-friendly way. User can change the configuration and re-run the computation to test out the hyperparameters easily.
7. Dynamically compute the input-output ports compatibility, dataframe columns names and types, ports types to prevent connection errors. 
8. Nodes can have multiple output ports that can be used to generate different output types. E.g. some data loader Node provides both `cudf` and `dask_cudf` output ports. The multiple GPUs distributed computation computation is automatically enabled by switching to the `dask_cudf` output port. 
9. Provides the standard API to extend your computation Nodes.
10. The composite node can encapsulate the TaskGraph into a single node for easy reuse. The composite node can be exported as a regular greenflow node without any coding.


This extension is composed of a Python package named `greenflowlab`
for the server extension and a NPM package named `greenflowlab`
for the frontend extension.


## Build the dev container

In the greenflow root direction
```bash
bash docker/build.sh
```
Launch your development container, make sure mounting your greenflow directory
to the container and open `8888` ports.

Set the greenflow path as the folder to start the development or you can open the 
`workspace.code-workspace` file.

## Requirements

* JupyterLab >= 3.0

## Install

```bash
pip install greenflowlab
```


## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```


## Contributing

### Development install

Note: You will need NodeJS to build the extension package.

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Change directory to the greenflowlab directory
# Install package in development mode
pip install -e .
# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Rebuild extension Typescript source after making changes
jlpm run build
```

You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
jlpm run watch
# Run JupyterLab in another terminal
jupyter lab
```

With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

By default, the `jlpm run build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:

```bash
jupyter lab build --minimize=False
```

### Uninstall

```bash
pip uninstall greenflowlab
```
