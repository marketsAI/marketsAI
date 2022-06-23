# marketsAI

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

marketsAI is a modular framework designed to simulate economies and markets. Each economy or market is an OpenAI Gym compatible environment.

## Installation

Because the library is still on heavy development we are not yet on PyPi. You can however install the package using pip:

```shell
pip install https://github.com/marketsAI/marketsAI/archive/alpha.zip
```

## Structure

The main code is in the `marketsai/` folder. The relevant sub-folers are:

`economies/`: In this folder there are two types of scripts:

- Scripts starting in run_ are desgined to run environments. There you change the environment, specify the configuration of the training and create custom metrics.

- Scripts starting in analysis_ take trained models and evaluate them on the eval_model of the environemnts. Here we create graphs as well.

## Economies

- Growth Model
- Stochastic Growth Model
- Krusell Smith
- Heterogenous entrepreneurs
- Townsend
