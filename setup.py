from setuptools import setup, find_packages
import pathlib

current_dir = pathlib.Path(__file__).parent.resolve()
long_description = (current_dir / "README.md").read_text(encoding="utf-8")


def parse_requirements(filename):
    with open(current_dir / filename) as requirements_file:
        requirements = requirements_file.readlines()
        # Remove all lines that are comments
        requirements = [
            line for line in requirements if not line.strip().startswith("#")
        ]
        # Remove pip flags
        requirements = [
            line for line in requirements if not line.strip().startswith("--")
        ]
        # Remove inline comments
        requirements = [
            line.split("#", 1)[0] if "#" in line else line for line in requirements
        ]
        # Remove empty lines
        requirements = list(filter(None, requirements))
        # Remove whitespaces
        requirements = [line.strip().replace(" ", "") for line in requirements]
        return requirements


with open(
    current_dir / "city_modeller/VERSION",
    "r",
    encoding="utf-8",
) as vf:
    version = vf.read().strip()

setup(
    name="city_modeller",
    version=version,
    description="city_modeller_app",
    long_description=long_description,
    author="CEEU - UNSAM",
    author_email="Ceeu.eeyn@unsam.edu.ar",
    url="https://github.com/CEEU-lab/city_modeller",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(include=["city_modeller", "city_modeller.*"]),
    test_suite="tests",
    tests_require=["pytest>=4.6.3"],
    package_data={
        "city_modeller": ["VERSION"],
    },
    install_requires=[parse_requirements("requirements.txt")],  # TODO: Change to a .in
)
