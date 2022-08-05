# Federated Learning Example

This project provides a step-by-step guide for doing **federated learning** with a **PyTorch LSTM neural network** using the [**NVFLARE** framework](https://github.com/NVIDIA/NVFlare). Federated learning enables us to train collaboratively across multiple different machines. However to get you up and running as quickly as possible this example only requires a single machine and simulates the federated system locally. We do this using NVFLARE POC mode and simulate a federated learning environment with 1 server and 2 clients.

It is adapted from the `hello-pt` example on the NVFLARE GitHub repository but differs in that 1) it uses two different local datasets, 2) a custom model, 3) each client is able to train on it's own GPU, and 4) it uses **differential privacy**. 

All our FL code can be found in the `train-tabformer` folder. Inside you'll see two subfolders:
* custom: contains the custom components - here you'll find our model, our PyTorch dataset, our trainer, validator, etc.
* config: contains two json files specifying the client and server configurations.

Below you'll find a step-by-step guide on how to setup and run this project.

## Instructions

First, you'll need to setup the data and virtual environment following the steps in the file [`0_Setup.md`](/0_Setup.md).

Then, to run the example, you'll follow the steps in [`1_Run_Demo.md`](/1_Run_Demo.md).

-----




