# Building Documentation

A python environment that supports gQuant is required to build the docs. For
example:

```
export PYTHON_VERSION=3.6
export ENVNAME=py36-rapids

conda create -n $ENVNAME python=${PYTHON_VERSION}

conda install -n $ENVNAME -y \
    -c defaults -c nvidia -c rapidsai -c pytorch -c numba -c conda-forge \
    cudf=0.7 dask-cudf=0.7

# requirements for gquant
conda install -n $ENVNAME -y -c conda-forge bqplot PyYaml networkx
```

## Get additional dependencies

```bash
conda install -n $ENVNAME -y -c anaconda -c conda-forge \
    sphinx sphinx_rtd_theme recommonmark numpydoc
```

## Run makefile:

Clean/remove contents of the build sub-directory. Run `sphinx-apidoc` to
automatically generate documentation. Then generate html.

```bash
source activate $ENVNAME

# docs should be the current directory
rm -r ./build/*
sphinx-apidoc --module-first --ext-autodoc -f -o source/ ../gquant/
make html
```

Outputs to `build/html/index.html`.

