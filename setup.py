from setuptools import find_packages, setup
from booking import __version__

setup(
    name='telco_churn',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['wheel'],
    version=__version__,
    description='Demo repository implementing an end-to-end MLOps ie by deploying a classification workflow on Databricks.',
    authors='Edara Lakshmi Srinivas'
)