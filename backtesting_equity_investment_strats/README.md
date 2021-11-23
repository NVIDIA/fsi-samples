# How to Use GPUs for More Accurate Backtesting of Equity Investment Strategies [S32407]
High performance computing is important in investment back-testing when the number of scenarios is large. We'll introduce the Sharpe Ratio as a measure of investment success as well as bootstrapping, a statistical technique, to gain a robust number of potential market scenarios. For both, we will learn how to GPU-accelerate Python market simulation code using special packages like RAPIDS for data frames and Numba for numeric calculations. You'll benefit from this talk if you're not already familiar with how GPUs can be applied to HPC finance simulations.

## Features
1. The Python Jupyter Notebook code (Part1.2021.ipynb and Part2.2021.ipynb) and Dockerfile (which will access R and Python scripts for downloading datasets needed for GTC21 session) and requirements file are all contained here.<br>

## Build
docker build --network host --file docker/Dockerfile --tag beis:nvidia .

## Run 
docker run --network host --runtime=nvidia --gpus all --rm -it beis:nvidia

if there is the error: docker: Error response from daemon: Unknown runtime specified nvidia.
then refer to this page: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
under where you first see Setting up NVIDIA Container Toolkit and be sure to restart jupyter notebook.

## Run notebooks: already in Dockerfile
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''

## Release Notes
**v21.04**
1. First release (GTC21) <br>

## End User License Agreement (EULA)
Refer to the included Apache 2.0 License Agreement in **LICENSE** for guidance.

## References
[1] RDocumentation 2.0, https://github.com/datacamp/rdocumentation-2.0 <br>
[2] Bennett, M.J., Hugen, D.L., Financial Analytics with R: Building a Laptop Laboratory for Data Science, Cambridge University Press, 978-1316584460, 2016.

**Mark Bennett** mbennett@nvidia.com
