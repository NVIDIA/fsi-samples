#!/bin/bash
DATA_PATH=./check_points
mkdir -p $DATA_PATH
mkdir -p $DATA_PATH/512/
wget https://query.data.world/s/fb3ilrt77qcpx7kwnfgr3cybvdctk2 -O $DATA_PATH/model_best.pth.tar
wget https://query.data.world/s/o2kzs74pg22mc2mfyhkykyu6pq36yr -O $DATA_PATH/512/model_best.pth.tar
