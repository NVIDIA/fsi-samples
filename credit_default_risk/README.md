# Credit Default Risk 
Inspired by the Bank of England paper 816 entitled "Machine learning explainability in finance: an application to default risk analysis" [1]from August 2019 where a U.K. mortgage dataset was used, we use the widely available public U.S. Fannie Mae mortgage dataset to show an XGBoost classifer for predicting loan delinquencies. For Fannie Mae, there are the loan acquisition files (kept in subdir acq) and loan performance files (kept in subdir perf). The NVIDIA developer blog article is entitled "Explaining and Accelerating Machine Learning for Loan Delinquencies" by Mark Bennett, John Ashley, and Patrick Hogan [3]. Python code began with Kyle DeGrave's article entitled "Predicting Loan Defaults in the Fannie Mae Data Set" [2], and further code contributions were made for acceleration, deep learning, and explainability by Emanuel Scoullos, Jochen Papenbrock, and Mark Bennett of NVIDIA. More detailed analysis of explanability approaches for portfolio construction and mortgage loans is provided in the more recent articles [4], [5], [6] by Emanual Scoullos, Jochen Papenbrock, Prabhu Ramamoorthy, Thomas Schoenemeyer, and Miguel Martinez.

## Features
Revised code for four Python Jupyter Notebooks:

- `1_mortcudf_data_prep.ipynb`
- `2_mortcudf_XGB_Pytorch.ipynb`
- `3_mortcudf_captum.ipynb`
- `4_mortcudf_shapley_viz.ipynb`

with RAPIDS ETL code, XGBoost classifier code, SHAP value code, PyTorch classifier code, Captum code, Shapley Clustering and Visualization, and the Dockerfile are contained here. In addition, the original notebook to match the article from November 2020 is here and titled `mortcudf_xgb.ipynb`.

Note that docker command below should be run just above the docker directory.

## Build
The first step is to download the dataset by running the `download_data.sh` script. Downloading the data takes time as the dataset consumes up to <strong>195GB</strong> of space to wget, untar, then finally rm unnecessary perf files. Afterwards, the docker container can be built:

```
docker build -t cdr:nvidia --network host -f docker/Dockerfile .
```
If your home directory is running low on space, you can "Remove all unused containers, networks, images (both dangling and unreferenced), and optionally, volumes" using `docker system prune`. Please refer to the [Docker documentation](https://docs.docker.com/engine/reference/commandline/system_prune/)

Ensure that that PyTorch is installed with CUDA 11.1 or greater. This can be checked in the running container by running: `conda list torch`
A satisfactory output will be `1.9.0+cu111` for Pytorch 1.9.0 with CUDA 11.1. Otherwise please pip install a later version of Pytorch: `pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`

## Run

```
docker run --gpus all --rm -it -p 8888:8888 -p 8889:8889 -p 8890:8890 -p 8891:8891 -p 8005:8005 -v `pwd`:$HOME cdr:nvidia 
```

If there is the error: docker: Error response from daemon: Unknown runtime specified nvidia.
then refer to this page: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
under where you first see Setting up NVIDIA Container Toolkit and be sure to restart jupyter notebook.

## Run notebooks: already in Dockerfile
```
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
```

## Release Notes
**v21.06**
1. First release


## End User License Agreement (EULA)
Refer to the included Apache 2.0 License Agreement in **LICENSE** for guidance.

## References

[1] Bracke, P., Datta, A., Jung, C., and Sen, S., Bank of England working paper #816, Machine Learning Explainability in Finance an Application to Default Risk Analysis, https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2019/machine-learning-explainability-in-finance-an-application-to-default-risk-analysis.pdf <br>

[2] DeGrave, K., Predicting Loan Defaults in the Fannie Mae dataset, https://degravek.github.io/project-pages/project1/2016/11/12/New-Notebook/ <br>

[3] Bennett, M., Ashley, J., Hogan, P., Explaining and Accelerating Machine Learning for Loan Delinquencies, https://developer.nvidia.com/blog/explaining-and-accelerating-machine-learning-for-loan-delinquencies/ <br>

[4] Papenbrock, J., Bennett, M., Schoenemeyer, T., Scoullos, E., Martinez, M., and Ashley, J., Accelerating Trustworthy AI for Credit Risk Management, https://developer.nvidia.com/blog/accelerating-trustworthy-ai-for-credit-risk-management/ <br>

[5] Papenbrock, J., Accelerating Interpretable Machine Learning for Diversified Portfolio Construction, https://developer.nvidia.com/blog/accelerating-interpretable-machine-learning-for-diversified-portfolio-construction/ <br>

[6] Scoullos, E., Bennett, M., Ashley, J., Hogan, P., Ramamoorthy, P., Papenbrock, J., and Martinez, M., Deep Learning vs Machine Learning Challenger Models for Default Risk with Explainability, https://developer.nvidia.com/blog/deep-learning-vs-machine-learning-challenger-models-for-default-risk-with-explainability/ <br>
