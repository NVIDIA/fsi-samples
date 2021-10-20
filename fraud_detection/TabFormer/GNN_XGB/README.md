# Graph Neural Networks for Fraud Detection

In these example notebooks we show how one can GPU-accelerate the training of Graph Neural Networks for FSI uses-cases like Credit Card Transaction Fraud Detection. We show how one can train GNN in self-supervised way on transaction graph and then use those GNN embeddings as additional features in downstream XGBoost Fraud Detection model to possibly see a lift in accuracy. The intuition for using the GNN embeddings as features in downstream fraud model is the potential benefit of using graph intelligence to detect complex fraud patterns that exploit the graph structure. Since these GNN embeddings are trained in unsupervised way they can also be used in other downstream business use-cases like recommender systems, marketing etc.

The libraries we use include RAPIDS cuDF, Dask, cuML, DGL (Deep Graph Library) and XGBoost and the dataset is [an open source credit card transactions dataset from IBM](https://github.com/IBM/TabFormer#credit-card-transaction-dataset) consisting of 24 million transactions.

## About the Notebooks
* In Notebook 1a we show how RAPIDS cuDF can be used to accelerate the data preprocessing required to turn tabular transaction data into DGL Heterogeneous graph made up of Cards and Merchants. This notebook requires single GPU with at least 16GB memory.
* In Notebook 1b we show a multi-GPU version of Notebook 1a using Dask cuDF. And it requires at least two 16GB GPUs to run.
* In Notebook 2 we show single-GPU training of Graph Neural Network in DGL for Link Prediction Task (Self-Supervised Learninig) on the transaction graph we created in Notebook 1a. The notebook requires at least one 16GB GPU to run. In the associated script we show multi-GPU training in DGL.
* In Notebook 3a we show one can GPU-accelerate the data cleaninig and feature engineering required for training a XGBoost model for fraud detection on the TabFormer dataset. This notebook requires at least 1 GPU with minimum 20GB memory.
* In Notebook 3b we add GNN embeddings as additional features to the ones we engineered from TabFormer dataset in Notebook 3a. This notebook requires at least two 16GB GPUs to run.
* In Notebook 4a we train the XGBoost model on features engineered from the TabFormer dataset on  multiple GPUs. This notebook requires at least 40GB GPU memory total.
* In Notebook 4b we train the XGBoost model on GNN embeddings + features engineered from the TabFormer dataset. This notebook requires at least 120GB GPU memory (for example four V100 32GB GPUs).

## Steps to run the notebooks

### Run/build the container. 
If you are able to pull and have access to NGC container nvcr.io/nvidian/pytorch_dgl_rapids then just run `./run_docker_ngc.sh`

If not then use `build_docker.sh` to build the container then run it with `./run_docker_local.sh.` In both cases specify the GPUs to expose to the container. For example, to run on 1st two GPUs use `./run_docker_local.sh 0,1`

### Start Jupyter
Start Jupyterlab inside the container by running `./start_jupyter.sh`. That will launch the notebook at port `8888` (you can change this) so you can access it at `locahost:8888` or `YOUR_IP_ADDRESS:8888` depending on your setup.




