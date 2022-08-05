# Run the FL Demo

Before we can run our custom federated learning app, we need to first start the POC mode that simulates the FL environment. Next, we'll walk through how we start the federated training, how we monitor the progress, and finally how we shut down the system when it finishes.

**Note: This example was designed to be run on a machine with 2 GPUs available.** Each site trains on it's own GPU, however the central server doesn't need a GPU to aggregate the weights. By default it will use your first GPU to train site 1, and the second to train site 2. If you need to modify this behavior, this can be changed in the `_initialize_trainer()` function in our [trainer](/train-tabformer/custom/tabformer_lstm_trainer.py) and the `_initialize_validator()` function in the [validator](/train-tabformer/custom/tabformer_lstm_validator.py).

---

### 0. Setting up the application environment in POC mode
NVFLARE provides a "POC mode" that allows us to simulate the FL environment on a single local machine. Run the `poc` command to generates a poc folder with a server, two clients, and one admin:

```bash
poc -n 2
```

Copy necessary files (our `train-tabformer` folder) to a working folder (the upload folder for the admin). Note that we put all our necessary files in the `transfer` directory of the admin client (which will then upload the files to the server):

```bash
mkdir -p poc/admin/transfer
cp -rf train-tabformer poc/admin/transfer
```

---

### 1. Start the application environment in POC mode
Once you are ready to start the FL system, you can run the following commands to start all the different parties. 

For ease of use, I recommend opening four terminals at this point. In each terminal ensure that you are in the main project directory and **that your virtual environment with NVFLARE installed is activated**.

We'll start the server first. Once the server is running you can start the clients in different terminals.

**In your first terminal, start the server:**
```bash
./poc/server/startup/start.sh
```

**In the second terminal, we'll start the first client (i.e. site-1):**
```bash
./poc/site-1/startup/start.sh
```

**In the third terminal, we'll start the second client (i.e site-2):**
```bash
./poc/site-2/startup/start.sh
```

**And in the fourth terminal, we'll start the admin:**
```bash
./poc/admin/startup/fl_admin.sh localhost
```

This will launch a command prompt where you can input admin commands to control and monitor many aspects of the FL process. Log in by entering `admin` for both the username and password.

---

### 2. Train federated! or running the FL System
Now you can use admin commands to upload, deploy, and start this example app. With the admin client command prompt successfully connected and logged in, enter the commands below in order. Pay close attention to what happens in each of four terminals. You can see how the admin controls the server and clients with each command.


Upload the application from the admin client to the server’s staging area:
```bash
upload_app train-tabformer
```

Create a run directory in the workspace for the run_number on the server and all clients. The run directory allows for the isolation of different runs so the information in one particular run does not interfere with other runs:
```bash
set_run_number 1
```
This will make the train-tabformer application the active one in the run_number workspace. After the following command and the next command, the server and all the clients know the train-tabformer application will reside in the run_1 workspace:

```bash
deploy_app train-tabformer all
```
This start_app command instructs the NVIDIA FLARE server and clients to start training with the train-tabformer application in that run_1 workspace:
```bash
start_app all
```

From time to time, you can issue `check_status server` in the admin client to check the entire training progress.

You should now see how the training does in the very first terminal (the one that started the server).


---

### 3. Shutdown the system

Once the fl run is complete and the server has successfully aggregated the clients’ results after all the rounds, run the following commands in the fl_admin to shutdown the system (while inputting admin when prompted with user name):

```bash
shutdown client
shutdown server
```

---

### Examine the results
All artifacts from the FL run can be found in the server run folder you created with set_run_number. In this exercise, the folder is run_1.

---

#### Congratulations! You’ve successfully built and run your first federated learning system. 
