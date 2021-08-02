# marketsAI

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

marketsAI is a modular framework designed to simulate economies and markets. Each economy or market is an OpenAI Gym compatible environment.

## Installation

Because the library is still on heavy development we are not yet on PyPi. You can however install the package using pip:
```shell
pip install https://github.com/marketsAI/marketsAI/archive/alpha.zip
```

## Folders

The main cofe bas is the marketsai folder. The relevant sub-folers are:

documents: It contain the latex of the related documents (papers, slides, etc.) and the figures that go on those documents.

experiments: In this folder there are two types of srcits. 
    Scripts starting in run_ are desgined to run environments. There you change the environment, specify the configuration of the training and create custom metrics.
    Scripts starting in analysis_ take trained models and evaluate them on the eval_model of the environemnts. Here we create graphs as well.

economies: here we specify economies as environments that are suitable for RL training. Write know there are two folders, one with single-agent environemnts and one wil multi-agent environments. 

econ_algos: here we have the scripts that solve models of the economies using thradition global methods (mostly using the GDSGE package).

## To do:

Put functions into utils. Justo one script with every auxiliary function.
Put experiments logger into the folders. 



