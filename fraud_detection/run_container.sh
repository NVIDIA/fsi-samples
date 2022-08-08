#/bin/bash

# for the impatient ...
docker build -t fsi_fd:nvidia .

docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --network host fsi_fd:nvidia bash

