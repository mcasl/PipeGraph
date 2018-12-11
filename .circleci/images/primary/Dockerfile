FROM circleci/python:3.6
COPY /requirements.txt $HOME/
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
   && python3 -m venv $HOME/venv  \
   &&  . $HOME/venv/bin/activate  \
   && pip install --upgrade pip \
   && pip install -U -r requirements.txt \
