# Setup Instructions

These instructions will show how to download the dataset, obtain the dataset splits, configure the dataset path and create the virtual environment for running the example.

---

### 0. Download the dataset
The TabFormer dataset can be downloaded directly from IBM's box website here: https://ibm.ent.box.com/v/tabformer-data/folder/130747715605

We've created a empty directory called `data` where you can place the dataset once downloaded.

---

### 1. Virtual environment and requirements
Next, you'll create the Python virtual environment and install the necessary libraries.

```bash
# create a new virtual env called nvflare-env
python3 -m venv nvflare-env

# activate virtual env
. nvflare-env/bin/activate

# install requirements
python3 -m pip install -r requirements.txt 
```

---

### 2. Generate data splits (do this once)
Once downloaded, you'll need to generate the two splits of the data corresponding to each client's local data. 
In this example I have split the dataset based on region - i.e. assuming client 1 and client 2 collect the same type of data but in different geographic regions. **In order to generate your own splits run the notebook `clean_split_tabformer.ipynb`.**

To start up the Jupyter notebook, run the following command:
```
jupyter notebook --ip 0.0.0.0 --no-browser
```

At the end of the notebook you'll see that the resulting data splits are named `site-1.csv` and `site-2.csv` and are automatically saved to our `data` folder. In order for the FL example to run you'll need to use the same file names.

---

### 3. Check path in config
Update the `dataset_base_dir` argument in the client config file (`train-tabformer/config/config_fed_client.json`) to match your local path to datasets folder. 

**Note:** Specify the **absolute** path to the `data` folder in the config. i.e. do not include relative paths such as `~`

---
