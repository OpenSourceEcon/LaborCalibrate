'''
------------------------------------------------------------------------
This module contains utility functions that do not naturally fit with
any of the other modules.

This Python module defines the following function(s):
    print_time()
    compare_args()
    create_graph_aux()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def print_time(seconds, type):
    '''
    --------------------------------------------------------------------
    Takes a total amount of time in seconds and prints it in terms of
    more readable units (days, hours, minutes, seconds)
    --------------------------------------------------------------------
    INPUTS:
    seconds = scalar > 0, total amount of seconds
    type    = string, either "SS" or "TPI"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    secs = scalar > 0, remainder number of seconds
    mins = integer >= 1, remainder number of minutes
    hrs  = integer >= 1, remainder number of hours
    days = integer >= 1, number of days

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Nothing
    --------------------------------------------------------------------
    '''
    if seconds < 60:  # seconds
        secs = round(seconds, 4)
        print(type + ' computation time: ' + str(secs) + ' sec')
    elif seconds >= 60 and seconds < 3600:  # minutes
        mins = int(seconds / 60)
        secs = round(((seconds / 60) - mins) * 60, 1)
        print(type + ' computation time: ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
    elif seconds >= 3600 and seconds < 86400:  # hours
        hrs = int(seconds / 3600)
        mins = int(((seconds / 3600) - hrs) * 60)
        secs = round(((seconds / 60) - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(hrs) + ' hrs, ' +
              str(mins) + ' min, ' + str(secs) + ' sec')
    elif seconds >= 86400:  # days
        days = int(seconds / 86400)
        hrs = int(((seconds / 86400) - days) * 24)
        mins = int(((seconds / 3600) - days * 24 - hrs) * 60)
        secs = round(
            ((seconds / 60) - days * 24 * 60 - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(days) + ' days, ' +
              str(hrs) + ' hrs, ' + str(mins) + ' min, ' +
              str(secs) + ' sec')


def compare_args(contnr1, contnr2):
    '''
    --------------------------------------------------------------------
    Determine whether the contents of two tuples are equal
    --------------------------------------------------------------------
    INPUTS:
    contnr1 = length n tuple
    contnr2 = length n tuple

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    len1     = integer >= 1, number of elements in contnr1
    len2     = integer >= 1, number of elements in contnr2
    err_msg  = string, error message
    same_vec = (len1,) boolean vector, =True for elements in contnr1 and
               contnr2 that are the same
    elem     = integer >= 0, element number in contnr1
    same = boolean, =True if all elements of contnr1 and contnr2 are
           equal

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: same
    --------------------------------------------------------------------
    '''
    len1 = len(contnr1)
    len2 = len(contnr2)
    if not len1 == len2:
        err_msg = ('ERROR, compare_args(): Two tuples have different ' +
                   'lengths')
        print(err_msg)
        same = False
    else:
        same_vec = np.zeros(len1, dtype=bool)
        for elem in range(len1):
            same_vec[elem] = np.min(contnr1[elem] == contnr2[elem])
        same = np.min(same_vec)

    return same


def create_graph_aux(vector, vector_name):
    '''
    --------------------------------------------------------------------
    Plot steady-state equilibrium results
    --------------------------------------------------------------------
    INPUTS:
    vector = (S,) vector, steady-state vector to be plotted

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    S           = integer in [3, 80], number of periods an individual
                  lives
    vector_full   = (S+1,) vector, vector with zero appended on end.
    age_pers_b  = (S+1,) vector, ages from 1 to S+1

    FILES CREATED BY THIS FUNCTION:
        vector_name.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # Plot steady-state consumption and savings distributions
    S = vector.shape[0]
    vector_full = np.append(vector, 0)
    age_pers_b = np.arange(1, S + 2)
    fig, ax = plt.subplots()
    plt.plot(age_pers_b, vector_full, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.xlim((20, S + 1))
    output_path = os.path.join(output_dir, vector_name)
    plt.savefig(output_path)
    # plt.show()
    plt.close()


def create_graph_calib(vector, vector_name):
    '''
    --------------------------------------------------------------------
    Plot calibration data results
    --------------------------------------------------------------------
    INPUTS:
    vector = (S,) vector, calibration vector to be plotted
    vector_name = string, name of image to be saved

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    images_fldr = string, folder in current path to save files
    images_dir  = string, total path of images folder
    images_path = string, path of file name of figure to be saved
    S           = integer in [3, 80], number of periods an individual
                  lives
    age_pers    = (S,) vector, ages from 1 to S

    FILES CREATED BY THIS FUNCTION:
        vector_name.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    images_fldr = 'images/calibration'
    images_dir = os.path.join(cur_path, images_fldr)
    if not os.access(images_dir, os.F_OK):
        os.makedirs(images_dir)

    # Plot steady-state consumption and savings distributions
    S = vector.shape[0]
    age_pers = np.arange(21, 21 + S)
    fig, ax = plt.subplots()
    plt.plot(age_pers, vector, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.xlim((20, 20 + S + 1))
    images_path = os.path.join(images_dir, vector_name)
    plt.savefig(images_path)
    # plt.show()
    plt.close()
