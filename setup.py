from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()  

setup(
    name="MLOPS_Hotel_Reservation",
    version="0.1.0",
    author="Prashant H V",
    packages=find_packages(),
    install_requires=requirements,                                  
)              