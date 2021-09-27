## Greenflow Cusignal Plugin 

Greenflow plugin that includes a set of nodes for Cusignal library. 


### Install the greenflowlab JupyterLab plugin

First create a Python enviroment or use one with RAPIDS cuSignal library. Tip,
use mamba to resolve dependencies quicker.
```bash
conda create -n rapids_cusignal -c conda-forge mamba python=3.8

conda activate rapids_cusignal

mamba install -c rapidsai -c nvidia -c conda-forge \
    cusignal=21.06 python=3.8 cudatoolkit=11.2
```

Then install `greenflowlab` JupyterLab plugin, make sure `nodejs` of version
[12^14^15] is installed. E.g:
```bash
mamba install -c conda-forge python-graphviz nodejs=12.4.0 pydot
```
Then install the `greenflowlab`:
```bash
pip install greenflowlab
```
Or install `greenflowlab` at the greenflowlab directory:
```bash
pip install .
```

### Install the cusignal plugin
Install the plugin directly from the plugin diretory.
```bash
pip install .
```
