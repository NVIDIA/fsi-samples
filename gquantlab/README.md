# gquantlab

![Github Actions Status](https://github.com/rapidsai/gQuant/gquantlab/workflows/Build/badge.svg)

gQuant Jupyterlab extension


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

Note: You will need NodeJS to install the extension.

```bash
pip install gquantlab
jupyter lab build
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

### Start the JupyterLab

Once the gquantlab plugin is install, the jupyterlab can be started. There is 
one important environment to consider before starting. The custom module files 
are specified in the `gquantrc` file. You can find an example `gquantrc` file in
the gQuant root directory. `gquantrc` file is by default is read at the same location
as the jupyterlab server's root directory. However, this can be overwirtten by 
setting the `GQUANT_CONFIG` environment variable. In the example `gquantrc`, system 
environment variable `MODULEPATH` is used to point to the paths of the module files.
To start the jupyterlab, please make sure `MODULEPATH` is set properly. 

For example, if you want to start the jupyterlab in the gQuant root directory.
```bash
MODULEPATH=$PWD/modules jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```

Or, if you want to start the jupyterlab in the gquantlab directory.
```bash
GQUANT_CONFIG=../gquantrc MODULEPATH=$PWD/../modules jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```
