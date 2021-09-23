## Greenflow Plugin for Hierarchical Risk Parity(HRP) diversification

This package includes a set of Greenflow nodes[1] that accelerate the investment workflow in the GPU described in the paper[2]. It has following steps

* Load CSV data
* Run bootstrap to generate 1 million scenarios
* Compute assets distances to run hierarchical clustering and HRP weights for the assets
* Compute the weights for the assets based on naïve RP method
* Compute the Sharpe ratios difference between these two methods (HRP-NRP)
* Calculate features from assets return mean, std, drawdown, correlation. It also computes std, mean across assets and across yearly time slices.  It computes 30 features in total. 
* Use the features and target value  (the Sharpe ratio difference) to train a XGBoost model
* Run HPO to find out the best parameters for the XGBoost model
* Compute the Shap values from the XGBoost model and find out which feature explains the Sharpe difference via visualization

It leverage the Numba GPU kernel[3] to accelerate customized computation. Dask[4] is used to parallelize the Bootstrap sample computation in different GPUs.  


## How to install

### Method 1. Docker
In this project directory, build the docker image:
```bash
docker build --network=host -f docker/Dockerfile -t hrp_example .
```
Launch the container by:
```bash
docker run -it --rm -p8888:8888 --gpus all hrp_example
```
In case you have the data files in `/path/to/pricess.csv`, you can mount it when launching the container
```bash
docker run -it --rm -p8888:8888 -v/path/to/:/workspace/notebooks/data/ --gpus all hrp_example
```

### Method 2, Conda  Install
#### Create a new Python environment
```bash
conda create -n test python=3.8
conda activate test
```

#### Install the Greenflow 

To install the Greenflow graph computation library, run the following command:
```bash
pip install greenflow
```

#### Install the greenflowlab JupyterLab plugin
To install `greenflowlab` JupyterLab plugin, make sure `nodejs` of version [12^14^15] is installed. E.g:
```bash
conda install -c conda-forge python-graphviz nodejs=12.4.0 pydot
```
install `greenflowlab` by:
```bash
pip install greenflowlab
```

#### Install the latest RAPIDS
```bash
conda install -y -c rapidsai -c nvidia -c conda-forge rapids=21.06 cudatoolkit=11.0
```

#### Install the Greenflow RAPIDS plugin
Install `greenflow_gquant_plugin`:
```bash
pip install greenflow_gquant_plugin
```

#### Install the greenflow_hrp_plugin
To install this plugin, clone this repo first.  Run following command at the root directory of this project
```bash
pip install .
```

#### Run the examples
Launching the Jupyter Lab[5] by,
```bash
jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
```
Example notebooks are in the notebooks directory

#### Run unit tests
Run all the unit tests
```bash
python -m unittest tests/unit/test_*.py -v
```

## Make tar release
```bash
bash make_tar.sh
```

## References

1. https://github.com/NVIDIA/fsi-samples/tree/main/greenflow
2. Markus J, Stephan K et al. Interpretable Machine Learning for Diversified Portfolio Construction, The Journal of Financial Data Science Summer 2021, Jan 2021
3. https://numba.pydata.org/
4. https://dask.org/
5. http://jupyterlab.io/
