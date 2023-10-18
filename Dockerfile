FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

# Install relevant system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  sudo \
  gpg \
  software-properties-common \
  dirmngr \
  libatlas-base-dev \
  libgdal-dev \
  gfortran \
  git \
  && gpg --keyserver keyserver.ubuntu.com \
    --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' \
  && gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' | \
    sudo tee /etc/apt/trusted.gpg.d/cran_debian_key.asc \
  && add-apt-repository "deb http://cloud.r-project.org/bin/linux/debian buster-cran40/" \ 
  && apt-get install r-base r-base-dev r-recommended r-base-core

# R pckgs
RUN R -e "install.packages('raster', dependencies=TRUE, Ncpus=16)"
RUN R -e "install.packages(c('dplyr','magrittr','splines'), repos='https://cloud.r-project.org/', Ncpus=6)"

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install relevant pip packages
RUN pip3 install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

RUN R -e "install.packages(c('tidyverse'), repos='https://cloud.r-project.org/')"

COPY . .

RUN pip3 install -e .
RUN pip3 install -e .[pymeanshift]

ENTRYPOINT [ "streamlit", "run" ]
CMD ["/app/city_modeller/main.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
