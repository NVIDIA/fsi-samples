#!/bin/bash
ext=`date +"%y_%m.tgz"`
root=.
mandatory="$root/README.md $root/setup.py"
common="$root/greenflow_hrp_plugin $root/notebooks $root/docker"
#doc=$root/Documents
excl="--exclude=*/notebooks/data/pricess.csv --exclude=*/.ipynb_checkpoints/*  --exclude=*/notebooks/ray* --exclude=*/__pycache__* --exclude=*/dask-worker-space* --exclude=*/.*"
 
tar cvfz "Nvidia_FSI_MunichRe_v"$ext $excl $mandatory $common
