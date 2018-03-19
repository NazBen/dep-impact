#/usr/bin/env sh
# Inspired from https://github.com/NazBen/scikit-optimize/blob/master/build_tools/travis/install.sh
# Skip Travis related code on circle ci.
if [ -z $CIRCLECI ]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate
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
CONDA=$HOME/miniconda
bash miniconda.sh -b -p $CONDA
export PATH="$CONDA/bin:$PATH"
conda update --quiet --yes conda
popd

# Create a conda env and install packages
conda create -n testenv --quiet --yes python=$PYTHON_VERSION nose pip \
	matplotlib pandas h5py scikit-learn 

source activate testenv

pip install -q pyDOE scikit-optimize
conda install --quiet --yes -c conda-forge openturns readline
conda install --quiet --yes -c R rpy2 R-base R-copula R-rcpp R-doparallel R-rcpparmadillo

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

R -e 'install.packages("VineCopula", repos="https://cloud.r-project.org", quiet=TRUE)'

python setup.py install

python -c "import matplotlib.pyplot as plt"
