#!/bin/bash
docker build --no-cache -t pytorch_dgl_rapids --network host -f Dockerfile .
