# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name="reward-poisoned-drl",
    version="0.1dev",
    packages=find_packages(),
    long_description=open("README.md").read(),
)
