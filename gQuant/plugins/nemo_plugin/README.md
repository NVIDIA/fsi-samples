## NeMo Plugin Example

This is an example to show how to write an external greenflow plugin. greenflow take advantage of the `entry point` inside the `setup.py` file to register the plugin. greenflow can discover all the plugins that has the entry point group name `greenflow.plugin`. Check the `setup.py` file to see details.

### Create an new Python enviroment
```bash
conda create -n test python=3.8
```

### Install the greenflow 
To install the greenflow graph computation library, run:
```bash
pip install greenflow
```
Or install `greenflow` at the greenflow directory:
```bash
pip install .
```

### Install the greenflowlab JupyterLab plugin
To install `greenflowlab` JupyterLab plugin, make sure `nodejs` of version [12^14^15] is installed. E.g:
```bash
conda install -c conda-forge nodejs=12.4.0
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
It depends on `greenflow_gquant_plugin` plugin, install it first. Check the README file in `greenflow_gquant_plugin` directory.
Next install `nemo` library. Currently, it is only compatible with old version of nemo.
```
git clone -b v0.11.1 https://github.com/NVIDIA/NeMo.git
cd NeMo
cp ../nemo.patch .
git apply nemo.patch && bash reinstall.sh
```
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
