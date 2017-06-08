#/usr/bin/env sh
export PATH="$HOME/miniconda/bin:$PATH"
source activate testenv

if [ ${COVERAGE} == "true" ]; then
	nosetests --with-coverage --with-timer --timer-top-n 10; 
else
    nosetests --with-timer --timer-top-n 10;
fi
nosetests -v --with-timer
