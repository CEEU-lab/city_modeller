FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

# Install relevant system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  software-properties-common \
  libatlas-base-dev \
  libgdal-dev \
  gfortran \
  git
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install relevant pip packages
RUN pip3 install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN pip3 install -e .
RUN pip3 install -e .[pymeanshift]

ENTRYPOINT [ "streamlit", "run" ]
CMD ["/app/city_modeller/main.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]