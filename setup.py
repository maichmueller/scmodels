#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

extras_requirements = {"plot": ["pygraphviz", "graphviz"], "test": ["pytest"]}

author = "Michael Aichmueller"

# This directory
dir_setup = os.path.dirname(os.path.realpath(__file__))
__version__ = ""
with open(os.path.join(dir_setup, 'scmodels', 'version.py')) as f:
    # Defines __version__
    exec(f.read())

setup(
    author=author,
    author_email="m.aichmueller@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    description="Structural Causal Models",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords="bayesian graphs scm sem fcm causal graphical models causality",
    name="scmodels",
    packages=find_packages(exclude=("test",)),
    setup_requires=setup_requirements,
    test_suite="test",
    tests_require=test_requirements,
    extras_require=extras_requirements,
    url="https://github.com/maichmueller/scmodels",
    version=__version__,
    zip_safe=False,
)
