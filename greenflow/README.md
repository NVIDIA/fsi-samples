# greenflow - Graph Computation Toolkit

## What is greenflow?

greenflow is a tool that helps you to organize the workflows. 

1. It define a TaskGraph file format `.gq.yaml` that describes the workflow. It can be edited easily by `greenflowlab` JupyterLab plugin.
2. Dynamically compute the input-output ports compatibility, dataframe columns names and types, ports types to prevent connection errors. 
3. Nodes can have multiple output ports that can be used to generate different output types. E.g. some data loader Node provides both `cudf` and `dask_cudf` output ports. The multiple GPUs distributed computation computation is automatically enabled by switching to the `dask_cudf` output port. 
4. Provides the standard API to extend your computation Nodes.
5. The composite node can encapsulate the TaskGraph into a single node for easy reuse. The composite node can be exported as a regular greenflow node without any coding.
6. greenflow can be extended by writing a plugin with a set of nodes for a particular domain. Check `plugins` for examples.

These examples can be used as-is or, as they are open source, can be extended to suit your environments.

## Binary pip installation

To install the greenflow graph computation library, run:
```bash
pip install greenflow
```
Or install `greenflow` at the root directory:
```bash
pip install .
```

greenflow node plugins can be registered in two ways: 

  1. (Recommended)Write a external plugin using 'entry point' to register it. Check the `external` directory for details
  2. Register the plugin in `greenflowrc` file. Check the `System environment` for details

## System environment 

There are a few system environment that the user can overwrite. 

The custom module files are specified in the `greenflowrc` file. `GREENFLOW_CONFIG` enviroment variable points to the location of this file. By default, it points to 
`$CWD\greenflowrc`. 

In the example `greenflowrc`, system environment variable `MODULEPATH` is used to point to the paths of the module files.
To start the jupyterlab, please make sure `MODULEPATH` is set properly. 
