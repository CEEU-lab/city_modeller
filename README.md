# city_modeller
Urban dynamics performance assessment with data-driven modelling templates

 Install the app in `Root` mode

```
virtualenv venv --python=python3.9
source venv/bin/activate
pip install -r requirements.txt

streamlit run urban_modeller.py
```
* Install the app in `Develop` mode

```
virtualenv venv --python=python3.9
source venv/bin/activate
python setup.py develop
pip install -r requirements-dev.txt

streamlit run urban_modeller.py

```