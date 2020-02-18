'''
------------------------------------------------------------------------
This module contains the functions that generate aggregate variables in
the steady-state or in the transition path of the overlapping
generations model with S-period lived agents and endogenous labor supply
from Chapter 7 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_L()
    get_K()
    get_Y()
    get_C()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_L(narr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate labor L or time path of aggregate
    labor L_t

    We have included a stitching function for L when L<=epsilon such
    that the the adjusted value is the following. Let sum(n_i) = X.
    Market clearing is usually given by L = X:

    L = X when X >= epsilon
      = f(X) = a * exp(b * X) when X < epsilon
                   where a = epsilon / e  and b = 1 / epsilon

    This function has the properties that
    (i) f(X)>0 and f'(X) > 0 for all X,
    (ii) f(eps) = eps (i.e., functions X and f(X) meet at epsilon)
    (iii) f'(eps) = 1 (i.e., slopes of X and f(X) are equal at epsilon)
    --------------------------------------------------------------------
    INPUTS:
    narr = (S,) vector or (S, T_S-1) matrix, values for steady-state
           labor supply (n1, n2, ...nS) or time path of the distribution
           of labor supply

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon = scalar > 0, small value at which stitch f(K) function
    a       = scalar > 0, scale parameter on stitching function
    b       = scalar > 0, coefficient on X in exponent of stitching
              function
    L       = scalar > 0, aggregate labor
    L_cstr  = boolean or (T+S-2,) boolean vector, =True if L <= epsilon
              or if L_t <= epsilon

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L, L_cstr
    --------------------------------------------------------------------
    '''
    epsilon = 0.1
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    if narr.ndim == 1:  # This is the steady-state case
        L = narr.sum()
        L_cstr = L < epsilon
        if L_cstr:
            print('get_L() warning: distribution of labor supply ' +
                  'and/or parameters created L < epsilon')
            # Force L > 0 by stitching a * exp(b * L) for L < eps
            L = a * np.exp(b * L)
    elif narr.ndim == 2:  # This is the time path case
        L = narr.sum(axis=0)
        L_cstr = L < epsilon
        if L.min() < epsilon:
            print('Aggregate labor constraint is violated ' +
                  '(L_t < epsilon) for some period in time path.')
            L[L_cstr] = a * np.exp(b * L[L_cstr])

    return L, L_cstr


def get_K(barr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate capital stock K or time path of
    aggregate capital stock K_t.

    We have included a stitching function for K when K<=epsilon such
    that the the adjusted value is the following. Let sum(b_i) = X.
    Market clearing is usually given by K = X:

    K = X when X >= epsilon
      = f(X) = a * exp(b * X) when X < epsilon
                   where a = epsilon / e  and b = 1 / epsilon

    This function has the properties that
    (i) f(X)>0 and f'(X) > 0 for all X,
    (ii) f(eps) = eps (i.e., functions X and f(X) meet at epsilon)
    (iii) f'(eps) = 1 (i.e., slopes of X and f(X) are equal at epsilon)
    --------------------------------------------------------------------
    INPUTS:
    barr = (S,) vector or (S, T+S-1) matrix, values for steady-state
           savings (b_1, b_2,b_3,...b_S) or time path of the
           distribution of savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon = scalar > 0, small value at which stitch f(K) function
    a       = scalar > 0, scale parameter on stitching function
    b       = scalar > 0, coefficient on X in exponent of stitching
              function
    K       = scalar or (T+S-2,) vector, steady-state aggregate capital
              stock or time path of aggregate capital stock
    K_cstr  = boolean or (T+S-2) boolean vector, =True if K <= epsilon
              or if K_t <= epsilon

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, K_cstr
    --------------------------------------------------------------------
    '''
    epsilon = 0.1
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    if barr.ndim == 1:  # This is the steady-state case
        K = barr.sum()
        K_cstr = K < epsilon
        if K_cstr:
            print('get_K() warning: distribution of savings and/or ' +
                  'parameters created K < epsilon')
            # Force K > 0 by stitching a * exp(b * K) for K < eps
            K = a * np.exp(b * K)

    elif barr.ndim == 2:  # This is the time path case
        K = barr.sum(axis=0)
        K_cstr = K < epsilon
        if K.min() < epsilon:
            print('Aggregate capital constraint is violated ' +
                  '(K_t < epsilon) for some period in time path.')
            K[K_cstr] = a * np.exp(b * K[K_cstr])

    return K, K_cstr


def get_Y(K, L, params):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate output Y or time path of aggregate
    output Y_t
    --------------------------------------------------------------------
    INPUTS:
    K      = scalar > 0 or (T+S-2,) vector, aggregate capital stock
             or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, aggregate labor or time
             path of the aggregate labor
    params = length 2 tuple, production function parameters
             (A, alpha)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0,1), capital share of income
    Y      = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
             time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def get_C(carr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    carr = (S,) vector or (S, T) matrix, distribution of consumption c_s
           in steady state or time path for the distribution of
           consumption

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    C = scalar > 0 or (T,) vector, aggregate consumption or time path of
        aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C
