# Calibrate Disutility of Labor (chi^n_s) in OG Model
This repository contains instruction and code for calibrating the `chi^n_s` level parameters on individual disutility of labor functions in the `S`-period-lived agent overlapping generations model with endogenous labor.


## Information about the model
A description of the model in this code is given in the PDF chapter, [`OGtext_ch7.pdf`](https://github.com/OpenSourceMacro/LaborCalibrate/blob/master/documentation/OGtext_ch7.pdf), in the documentation folder. The model solution and calibration are coded in Python using Python 3.6. The solution can be executed by navigating to the [`LaborCalibrate/code/`](https://github.com/OpenSourceMacro/LaborCalibrate/tree/master/code) directory and running the script [`execute.py`](https://github.com/OpenSourceMacro/LaborCalibrate/blob/master/code/execute.py).


## Calibration approach
A description of our calibration approach is given in the PDF document, [`CalibrationNotes.pdf`](https://github.com/OpenSourceMacro/LaborCalibrate/blob/master/documentation/CalibrationNotes.pdf), in the documentation folder. We use a generalized method of moments approach to estimate the `S` values of the `chi^n_s` parameters in the model using the `S` household Euler equations for labor supply. These equations can only estimate the `chi^n_s` parameters up to a common factor. We then estimate that factor in our steady-state computation by making average income in the model match average income in the data.
