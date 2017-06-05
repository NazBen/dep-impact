# Inspired from https://github.com/NazBen/scikit-optimize/blob/master/build_tools/travis/install.sh

deactivate

# Install conda using miniconda
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo

if [[ ! -f miniconda.sh ]]; then
	if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
		wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
	else
		wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
	fi
fi

bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update --yes conda
conda info -a
popd


# Create env and install packages
conda create -n testenv --yes python=$TRAVIS_PYTHON_VERSION pip nose numpy \
	scipy scikit-learn matplotlib pandas h5py scikit-learn
source activate testenv

conda install --yes -c conda-forge openturns
conda install --yes -c R R r-copula

wget -q https://cran.r-project.org/src/contrib/VineCopula_2.1.2.tar.gz
R CMD INSTALL VineCopula_2.1.2.tar.gz

pip install pyDOE scikit-optimize rpy2 nose-timer

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py install
