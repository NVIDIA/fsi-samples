#!/bin/bash
source riva/config.sh

# download  docker-compose
VERSION=$(curl --silent https://api.github.com/repos/docker/compose/releases/latest | grep -Po  '"tag_name": "\K.*\d')
curl -L https://github.com/docker/compose/releases/download/${VERSION}/docker-compose-$(uname -s)-$(uname -m) -o riva/docker-compose
chmod 777 riva/docker-compose

pushd riva
bash riva_init.sh
popd

### start to build the container for server
docker build --network=host -f docker/Dockerfile.riva -t client .
