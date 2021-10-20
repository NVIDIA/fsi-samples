# Fraud Detection

This repository showcases fraud detection modeling methods applied to the Paysim, AmlSim, and TabFormer datasets for payment fraud detection. We employ a variety of methods for Fraud detection including XGBoost, Multilayer Perceptron (MLP), and Graphical Neural Networks (GNNs). We also include explainability with the Shap library and dataloading with NVTabular

# Running

To run the notebooks, first build the docker container and run it. A convenience bash script `run_container.sh` can be run from the command line to build and run the container:

`$ ./run_container.sh`

Make sure that you have execution capabilities, else adjust with `chmod` command. 

After the container is built and running, navigate to `localhost:8888` in your browser to view the notebooks. You should also download the datasets for the notebooks you wish to run. See the `Datasets` section below. The `AmlSim` directory has a simple convenience script you can run to download and extract the data - make sure you have a `7-zip` package like `7z` installed in your system.


# Datasets:

- PaySim dataset: https://www.kaggle.com/ntnu-testimon/paysim1
- AMLSim dataset: https://github.com/IBM/AMLSim
- TabFormer dataset: https://ibm.ent.box.com/v/tabformer-data/folder/130747715605

# References:

Papers and Repositories:

- BibTeX @misc{AMLSim, author = {Toyotaro Suzumura and Hiroki Kanezashi}, title = {{Anti-Money Laundering Datasets}: {InPlusLab} Anti-Money Laundering DataDatasets}, howpublished = {\url{http://github.com/IBM/AMLSim/}}, year = 2021 }

- EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs https://arxiv.org/abs/1902.10191
Scalable Graph Learning for Anti-Money Laundering: A First Look https://arxiv.org/abs/1812.00076

TGN references:
- TGN original authors's repository: https://github.com/twitter-research/tgn
- DGL implementation: https://github.com/dmlc/dgl/tree/master/examples/pytorch/tgn

- TabFormer Repository: https://github.com/IBM/TabFormer along with the corresponding paper: https://arxiv.org/abs/2011.01843

- Shap: https://github.com/slundberg/shap/blob/master/docs/index.rst

- NVTabular: https://github.com/NVIDIA/NVTabular

