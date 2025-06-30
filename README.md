# example-detection-python-models

## Overview
This repository contains sample Python models for use with SAS Detection Architecture. See [SAS Detection Architecture product documentation](https://go.documentation.sas.com/doc/en/frdcdc/v_024/frdrs/titlepage.htm?fromDefault=) for how to import and use the models.

## Getting the Conda Environment
MLServer bring-your-own environment requires a conda environment packed into a tarball using conda-pack on the same architecture as runtime container. Here you can use any x86_64 linux docker image as `<x86_64_LINUX_IMAGE>`. We are using Miniforge3 as the conda distribution.
```
docker run --user root --rm -it -v $(pwd)/PYTEST_0200:/PYTEST_0200 --entrypoint=/bin/bash <x86_64_LINUX_IMAGE>

# install miniforge inside the container
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O
bash Miniforge3-Linux-x86_64.sh -b && export PATH=$PATH:/root/miniforge3/bin

cd /PYTEST_0200 && \
  conda env create -f environment.yml && \
  conda install -y conda-pack && \
  conda pack --ignore-missing-files -f -n pytest -o environment.tar.gz
```

## License
This project is licensed under the [Apache 2.0 License](LICENSE).

## Contributing
This is a sample repository, the maintainers are not currently accepting contributions.
