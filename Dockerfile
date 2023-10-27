FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

# Install relevant system packages
RUN apt-get update && \
    apt-get -y install sudo \
    gpg \
    wget

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  software-properties-common \
  dirmngr \
  libatlas-base-dev \
  libgdal-dev \
  gfortran \
  git 

# R config 4.3.1
RUN gpg --keyserver keyserver.ubuntu.com \
    --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'
RUN gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' | \
    sudo tee /etc/apt/trusted.gpg.d/cran_debian_key.asc

RUN add-apt-repository "deb http://cloud.r-project.org/bin/linux/debian buster-cran40/"

RUN apt-get update && apt-get install -y --no-install-recommends \
  r-base \  
  r-base-dev \
  r-recommended \
  r-base-core 

# R pckgs
RUN R -e "install.packages('raster', dependencies=TRUE, Ncpus=16)"
RUN R -e "install.packages(c('dplyr','magrittr','splines'), repos='https://cloud.r-project.org/', Ncpus=6)"

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Download parcel geoms to data folder
RUN wget -P /app/city_modeller/data/ https://storage.googleapis.com/python_mdg/city_modeller/data/caba_parcels_geom.shp

# Install relevant pip packages
RUN pip3 install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN pip3 install -e .
RUN pip3 install -e .[pymeanshift]

ENTRYPOINT [ "streamlit", "run" ]
CMD ["/app/city_modeller/main.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
