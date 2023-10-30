# city_modeller
Urban dynamics performance assessment with data-driven modelling templates

 Install the app in `Root` mode

## Installation

To install the app, it is recommended to create a virtual environment, as shown below:

``` shell
> virtualenv venv --python=python3.10
> source venv/bin/activate
```

Or, using `conda`'s wrapper for virtualenv:

``` shell
> conda create -n venv python=3.10
> conda activate venv
```

To install the package, in order to run the app, one must run:

``` shell
> python setup.py install
```

Or

``` shell
> pip install .
> pip install .[pymeanshift]
```

## Contributing

To contribute to the app, it is recommended to first install it in develop mode, using

``` shell
> python setup.py develop
```

Or

``` shell
> pip install -e .
> pip install -e .[pymeanshift]
```

To comply with the app's code style and linting configuration, it is extremely recommended to also install the development requirements:

``` shell
> pip install -r requirements-dev.txt
```

## Usage

To run the urban modeller app's streamlit server locally, after installation, use:

``` shell
streamlit run main.py
```

## Running from Docker

To use a Docker container, you first need to build it using:

``` shell
docker build . -t city_modeller
```

And then run it using:

``` shell
docker run -p 8501:8501 -v $PWD:/app city_modeller
```

In this way, you can run the whole app without installing the Python dependencies.

## Running using Docker Compose

To avoid building the image manually and exposing the port, simply run

``` shell
docker-compose up streamlit
```

## Macromodeller configuration

In order to run the urban valuator dashboard, the local environment must have the R-base `4.3.1` release installed. 
Also, the following packages. 

``` shell
install.packages(c('dplyr','maggritr','splines','raster'))
```

If the app is running from docker container, the installation is already configured in the `Dockerfile`.