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
```

## Contributing

To contribute to the app, it is recommended to first install it in develop mode, using

``` shell
> python setup.py develop
```

Or

``` shell
> pip install -e .
```

To comply with the app's code style and linting configuration, it is extremely recommended to also install the development requirements:

``` shell
> pip install -r requirements-dev.txt
```

## Usage

To run the urban modeller app's streamlit server locally, use:

``` shell
streamlit run main.py
```
