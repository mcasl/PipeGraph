FROM circleci/python:3.6

RUN set -ex  \
   && OS_packages=' \
		build-essential \
		python-dev \
		python-setuptools \
		libatlas-dev \
		libatlas3-base \
		build-essential \
		python-dev \
		python-setuptools \
		dvipng \
		texlive-latex-base \
		texlive-latex-extra \
	' \
   && sudo apt-get update \
   && sudo apt-get install -y --no-install-suggests --no-install-recommends $OS_packages \
   && sudo rm -rf /var/lib/apt/lists/* \
   \
   && python_modules=' \
                  numpy \
                  scipy \
                  matplotlib \
                  setuptools \
                  pytest \
                  coverage \
                  sphinx \
                  pillow  \
                  sphinx-gallery \
                  sphinx_rtd_theme \
                  networkx \
                  pandas \
                  cython \
                  numpydoc \
                  scikit-learn \
                  seaborn \
        ' \
   && python3 -m venv $HOME/venv  \
   &&  . $HOME/venv/bin/activate  \
   && pip install --upgrade pip \
   && pip install $python_modules \
