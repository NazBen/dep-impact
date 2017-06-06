#/usr/bin/env sh
# Inspired from https://github.com/NazBen/scikit-optimize/blob/master/build_tools/travis/install.sh
# Skip Travis related code on circle ci.
if [ -z $CIRCLECI ]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate
    PYTHON_VERSION=$TRAVIS_PYTHON_VERSION
else
	PYTHON_VERSION="2.7.12"
fi

# Install conda using miniconda
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]; then
	if [[ "$PYTHON_VERSION" == "2.7" ]]; then
		wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
	else
		wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
	fi
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update --quiet --yes conda
popd

# Create a conda env and install packages
conda create -n testenv --quiet --yes python=$PYTHON_VERSION nose pip \
	matplotlib pandas h5py scikit-learn rpy2 R R-copula R-cpp R-doparallel

source activate testenv

pip install -quiet pyDOE scikit-optimize
conda install --quiet --yes -c conda-forge openturns

R -e 'install.packages("VineCopula", repos="https://cloud.r-project.org")'

python setup.py install

python -c "import matplotlib.pyplot as plt"