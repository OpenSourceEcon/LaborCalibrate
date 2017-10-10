# Code for Calibration of Disutility of Labor parameters

This folder contains the code to calibrate the `S` disutility of labor parameters `chi^n_s` in the `S`-period overlapping generations model from Chapter 7, "S-period-lived Agents with Endogenous Labor" of the textbook, *Overlapping Generations Models for Policy Analysis: Theory and Computation*. This code also solves for the steady-state and transition path equilibrium solutions. The files needed to run the model are the following five Python scripts and modules:

* `execute.py`
* `SS.py`
* `TPI.py`
* `households.py`
* `firms.py`
* `aggregates.py`
* `elliputil.py`
* `utilities.py`

This code was writen using the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3.6. The folders `images` and `OUTPUT` will be created (overwritten) in the course of running your script. The `images` folder will contain the images created in the process of running your code, both steady-state and transition path equilibria. The `OUTPUT` folder will contain the Python objects as well as underlying parameters (pickle .pkl files) from the steady-state and transition path equilibria.
