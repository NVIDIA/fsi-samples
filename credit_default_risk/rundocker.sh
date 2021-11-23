#/bin/bash
docker build -t cdr:nvidia --network host -f docker/Dockerfile .
docker run --gpus all --rm -it -p 8888:8888 -p 8889:8889 -p 8890:8890 -p 8891:8891 -p 8005:8005 -v `pwd`:/rapids/notebooks/ cdr:nvidia bash
