Steps to run the GNN+XGBoost Fraud Detection notebooks

1. First navigate to GNN_XGB directory

2. Run/build the container
	1. If you are able to pull and have access to NGC container nvcr.io/nvidian/torch_rapids_dgl0.7 then just run `./run_docker_ngc.sh`
	2. If not then use `build_docker.sh` to build the container then run it with `./run_docker_local.sh`. In both cases specify the GPUs to expose to the container. For example, to run on 1st two GPUs use `./run_docker_local.sh 0,1`

2. Start Jupyterlab inside the container by running `./start_jupyter.sh`. That will launch the notebook at port 8321 so you can access it at `locahost:8321` or `YOUR_IP_ADDRESS:8321`  

References:

TabFormer dataset can be downloaded directly from  IBM's box website here: https://ibm.ent.box.com/v/tabformer-data/folder/130747715605

Original TabFormer GitHub repo is here: https://github.com/IBM/TabFormer along with the corresponding paper: https://arxiv.org/abs/2011.01843

