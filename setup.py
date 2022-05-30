#!/usr/bin/env python

"""Setup script for lotkavolterra_simulator
"""

__author__  = "Florent Leclercq"
__version__ = "1.0"
__date__    = "2022"
__license__ = "GPLv3"

import setuptools
import io

# Open README.md
def readme():
    with io.open('README.md', mode='r', encoding='ISO-8859-1') as f:
        return f.read()

# Open requirements.txt
def requirements():
    with io.open('requirements.txt', mode='r') as f:
        return f.read().splitlines()

# Setup
setuptools.setup(
    name="lotkavolterra_simulator",
    version="1.0",
    author="Florent Leclercq",
    author_email="florent.leclercq@polytechnique.org",
    description="A Lotka-Volterra Bayesian hierarchical model simulator",
    long_description=readme(),
    long_description_content_type='text/markdown',
    url="http://florent-leclercq.eu",
    packages=setuptools.find_packages(),
    install_requires=requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
