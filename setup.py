# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["sgdbandit", "sgdbandit.agents", "sgdbandit.environments", "sgdbandit.utils"]

package_data = {"": ["*"]}

install_requires = [
    "dill>=0.3.8,<0.4.0",
    "joblib>=1.3.2,<2.0.0",
    "matplotlib>=3.8.2,<4.0.0",
    "numpy>=1.26.3,<2.0.0",
    "pre-commit>=2.16.0,<3.0.0",
    "seaborn>=0.13.2,<0.14.0",
    "tqdm>=4.66.1,<5.0.0",
    "scipy",    
    "ipykernel"
]

setup_kwargs = {
    "name": "SGDBandit",
    "version": "0.1.0",
    "description": "Code for algorithm SGDBandit to deal with MAB with heavy tails",
    "author": "None",
    "author_email": "None",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.9",
}


setup(**setup_kwargs)
