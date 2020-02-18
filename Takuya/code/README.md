# Code for Chapter 7: "S-period-lived Agents with Endogenous Labor"

This folder contains the code to solve the model presented in Chapter 7, "S-period-lived Agents with Endogenous Labor" of the textbook, *Overlapping Generations Models for Policy Analysis: Theory and Computation*. The files needed to run the model are the following five Python scripts and modules:

* `execute.py`
* `SS.py`
* `TPI.py`
* `households.py`
* `firms.py`
* `aggregates.py`
* `elliputil.py`
* `utilities.py`

This code was writen using the [Anaconda distribution](https://www.continuum.io/downloads) of Python 3.7. The folders `images` and `OUTPUT` will be created (overwritten) in the course of running your script. The `images` folder will contain the images created in the process of running your code, both steady-state and transition path equilibria. The `OUTPUT` folder will contain the Python objects as well as underlying parameters (pickle .pkl files) from the steady-state and transition path equilibria.

Before running your computation of the model solution, you should open the file `execute.py` and adjust the parameters of the model. The main parameters are in lines 84-113 of `execute.py`. You may also want to update the assumption of the Frisch elasticity of labor supply in line 139 or the assumed initial distribution of savings (wealth) for the transition path solution in lines 289-294. The model can be run by navigating in your terminal to the folder where you have cloned your fork of this repository and typing `python execute.py`.

For the current baseline parameterization of the model with eighty-period-lived individuals (`S=80`), the steady-state solution takes between 26 seconds and 2 minutes depending on which solution method you use. However, the transition path solution takes just over 4.3 hours to run (88 time path iterations). The transition path solution can be sped up tremendously by parallelizing its computation.
