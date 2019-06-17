#!/bin/bash
DATA_PATH=./notebook/data
mkdir -p $DATA_PATH
wget https://query.data.world/s/qlidesn2wqntqjqie5nc7p3lsuvcon -O $DATA_PATH/security_master.csv.gz
wget https://query.data.world/s/i2xn3byzbx3msm4gebstjjkqusanpt -O $DATA_PATH/stock_price_hist.csv.gz
