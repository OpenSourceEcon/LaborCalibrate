'''
------------------------------------------------------------------------
This module contains utility functions that do not naturally fit with
any of the other modules.

This Python module defines the following function(s):
    print_time()
    compare_args()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

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
