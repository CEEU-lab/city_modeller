from setuptools import setup, find_packages
import pathlib

current_dir = pathlib.Path(__file__).parent.resolve()
long_description = (current_dir / 'README.md').read_text(encoding='utf-8')

setup(
    name="CityMachinerie",
    version="v1.0.0",
    description='city_modeller_app',
    long_description=long_description,
    author='CEEU - UNSAM',
    author_email='Ceeu.eeyn@unsam.edu.ar',
    url='https://github.com/CEEU-lab/city_modeller',
    classifiers=[
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'numpy >= 1.22.2',
        'pandas >= 1.4.1',
        'matplotlib >= 3.4.3',
    ]
)
