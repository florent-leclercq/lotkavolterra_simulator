# Lotka-Volterra simulator #

[![arXiv](https://img.shields.io/badge/astro--ph.CO-arxiv%3A2209.11057-B31B1B.svg?style=flat)](https://arxiv.org/abs/2209.11057)
[![GitHub version](https://img.shields.io/github/tag/florent-leclercq/lotkavolterra_simulator.svg?label=version)](https://github.com/florent-leclercq/lotkavolterra_simulator)
[![GitHub commits](https://img.shields.io/github/commits-since/florent-leclercq/lotkavolterra_simulator/v1.0.svg)](https://github.com/florent-leclercq/lotkavolterra_simulator/commits)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/florent-leclercq/lotkavolterra_simulator/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/lotkavolterra_simulator.svg)](https://badge.fury.io/py/lotkavolterra_simulator)
[![Website florent-leclercq.eu](https://img.shields.io/website-up-down-green-red/http/pyselfi.florent-leclercq.eu.svg)](http://pyselfi.florent-leclercq.eu/)

Simulator of the Lotka-Volterra prey-predator system, with demographic and observational noise and biases.

<img src="https://raw.githubusercontent.com/florent-leclercq/lotkavolterra_simulator/master/simulation.png" width="70%" align="center"></img>

## Installation ##

This is a standard, low-weight python package, written with python 3. It is packaged at https://pypi.org and can be installed using [pip](https://pip.pypa.io/en/stable/):

    pip install lotkavolterra-simulator

Alternatively it is possible to clone the Github repository and to install using:

    pip install .

## Documentation ##

The model is described in section III of [Leclercq (2022)](https://arxiv.org/abs/2209.11057). The jupyter notebook [simulations.ipynb](https://github.com/florent-leclercq/lotkavolterra_simulator/blob/main/simulations.ipynb) illustrates how to run the code and plot prey and predator theoretical and observed number functions.

This code has been designed to illustrate concepts in simulation-based inference. It is used in [pySELFI](https://github.com/florent-leclercq/pyselfi) from version 2.0.

Limited user-support may be asked from the main author, Florent Leclercq.

## Contributors ##

* Florent Leclercq, florent.leclercq@iap.fr

## Reference ##

To acknowledge the use of lotkavolterra_simulator in research papers, please cite the paper [Leclercq (2022)](https://arxiv.org/abs/2209.11057):

*Simulation-based inference of Bayesian hierarchical models while checking for model misspecification*<br/>
F. Leclercq<br/>
Proceedings of the <a href="https://maxent22.see.asso.fr/" target="blank">41st International Conference on Bayesian and Maximum Entropy methods in Science and Engineering (MaxEnt2022)</a>, 18-22 July 2022, Paris, France<br />
<a href="https://doi.org/10.3390/psf2022005004" target="blank">	Physical Sciences Forum <b>5</b>, 4 (2022)</a>, <a href="https://arxiv.org/abs/2209.11057" target="blank">arXiv:2209.11057</a> [<a href="https://arxiv.org/abs/2209.11057" target="blank">astro-ph.CO</a>] [<a href="https://ui.adsabs.harvard.edu/?#abs/2022arXiv220911057L" target="blank">ADS</a>] [<a href="https://arxiv.org/pdf/2209.11057" class="document" target="blank">pdf</a>]

    @ARTICLE{lotkavolterra_simulator,
        author = {{Leclercq}, Florent},
        title = "{Simulation-based inference of Bayesian hierarchical models while checking for model misspecification}",
        journal = {Physical Sciences Forum},
        volume = 5,
        pages = 4,
        doi = {10.3390/psf2022005004},
        keywords = {Statistics - Methodology, Astrophysics - Instrumentation and Methods for Astrophysics, Mathematics - Statistics Theory, Quantitative Biology - Populations and Evolution, Statistics - Machine Learning},
        year = 2022,
        month = sep,
        eid = {arXiv:2209.11057},
        pages = {arXiv:2209.11057},
        archivePrefix = {arXiv},
        eprint = {2209.11057},
        primaryClass = {stat.ME},
        }

## License ##

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. By downloading and using lotkavolterra_simulator, you agree to the [LICENSE](https://github.com/florent-leclercq/lotkavolterra_simulator/blob/main/LICENSE), distributed with the source code in a text file of the same name.
