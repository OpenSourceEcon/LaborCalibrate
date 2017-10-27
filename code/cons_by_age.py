'''
------------------------------------------------------------------------
This module defines functions that implement the calibration approach
to generating average consumption expenditures by age.

This module defines the following functions:
    get_c_s_interp()
------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

'''
------------------------------------------------------------------------
Read-in (create) the data
------------------------------------------------------------------------
avg_cons = (10,) vector, average consumption from CEX for specific age
           bins
age_cuts = (11,) vector, age cutoff boundaries for age bins in avg_cons
age_midp = (10,) vector, midpoint age in age bins associated with
           avg_cons
------------------------------------------------------------------------
'''
avg_cons = np.array([29000, 29200, 30373, 48087, 58784,
                     60524, 55892, 46757, 34382, 29000])
age_cuts = np.array([-1, 0, 5, 25, 35, 45, 55, 65, 75, 105,
                     106])
age_midp = np.array([-0.5, 2.5, 12.5, 30, 40, 50, 60, 70, 90,
                     105.5])

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


def get_c_s_interp(age_data, c_data, graph=False):
    '''
    --------------------------------------------------------------------
    This function returns a continuous interpolating function object and
    a factor that together represent the continuous consumer expenditure
    function of continuous age.

    c(s) = (factor_c * (c_tilde(s) - c_data.min())) + c_data.min()
    --------------------------------------------------------------------
    INPUTS:
    age_data = (N,) vector, age bin midpoints corresponding to c_data
    c_data   = (N,) vector, consumption expenditure data points
    graph    = boolean, =True if create and save scatterplot of data and
               interpolated function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    c_s_interp = function object, interpolating function from
                 si.interp1d, c_tilde(s)
    factor_c   = scalar > 0, factor by which to multiply interpolated
                 function to get continuous underlying function c(s)

    FILES CREATED BY THIS FUNCTION:
        images/DataInterp_CEX.png

    RETURNS: factor_c
    --------------------------------------------------------------------
    '''
    c_s_interp = si.interp1d(age_data, c_data, kind='cubic')
    factor_c = 1.1  # This skipped spot is for factor_c calculation

    if graph:
        '''
        ----------------------------------------------------------------
        Create scatter plot of data and interpolated function through
        the data
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        num_points  = integer, number of points to use in plotting
                      interpolated function
        age_fine    = (num_points,) vector, equally spaced points in
                      support of age
        cons_interp = (num_points,) vector, consumption values
                      corresponding to age_fine
        ----------------------------------------------------------------
        '''
        # Create directory if images folder does not already exist in
        # current directory
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Create the scatter plot with interpolated curves
        num_points = int(1000)
        age_fine = np.linspace(age_data.min(), age_data.max(),
                               num_points)
        cons_interp = c_s_interp(age_fine)

        fig, ax = plt.subplots()
        plt.scatter(age_data, c_data, s=70, c='blue', marker='o',
                    label='Data')
        plt.plot(age_fine, cons_interp, c='green', linestyle='--',
                 label='Cubic spline $c_{tilde}(s)$')
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Average consumption expenditure by age: data ' +
                  'and interpolation', fontsize=15)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Consumption $c(s)$')
        x_axis_min = age_data.min() - 0.05 * (age_data.max() -
                                              age_data.min())
        x_axis_max = age_data.max() + 0.05 * (age_data.max() -
                                              age_data.min())
        plt.xlim((x_axis_min, x_axis_max))
        y_axis_min = c_data.min() - 0.05 * (c_data.max() - c_data.min())
        y_axis_max = c_data.max() + 0.05 * (c_data.max() - c_data.min())
        plt.ylim((y_axis_min, y_axis_max))
        plt.ylim((0.95 * c_data.min(), 1.05 * c_data.max()))
        plt.legend(loc='upper right')
        plt.text(x_axis_min, 18000, 'Source: Consumer Expenditure ' +
                 'Survey (CEX), 2013 Summary Data.', fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        figname = 'images/DataInterp_CEX'
        plt.savefig(figname)
        print('Saved figure: ' + figname + '.png')
        # plt.show()
        plt.close()

    return factor_c
