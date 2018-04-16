'''
------------------------------------------------------------------------
This module contains the functions that generate aggregate variables in
the steady-state or in the transition path of the OG-ITA overlapping
generations model with S-period lived agents, endogenous labor supply,
and heterogeneous ability.

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


def get_L(n_arr, args):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate labor L or time path of aggregate
    labor L_t.

    We have included a stitching function for L when L<=epsilon such
    that the the adjusted value is the following. Let
    sum(lambda_j*e_{j,s}*n_{n,s,t}) = X. Market clearing is usually
    given by L = X:

    L = X when X >= epsilon
      = f(X) = a * exp(b * X) when X < epsilon
                   where a = epsilon / e  and b = 1 / epsilon

    This function has the properties that
    (i) f(X)>0 and f'(X) > 0 for all X,
    (ii) f(eps) = eps (i.e., functions X and f(X) meet at epsilon)
    (iii) f'(eps) = 1 (i.e., slopes of X and f(X) are equal at epsilon)
    --------------------------------------------------------------------
    INPUTS:
    n_arr = (S, J) matrix or (S, J, T2+S) array, values for steady-state
            labor supply (n_{j,s}) or time path of the distribution of
            labor supply (n_{j,s,t})
    args  = length 3 tuple, (emat, lambdas, omegas)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    emat       = (S, J) matrix, e_{j,s} ability by age and ability type
    lambdas    = (J,) vector, income percentiles for ability types
    omegas     = (S,) vector or (S, Tpers) matrix, steady-state
                 population distribution (percents) by age or time path
                 of population distribution. Time path: t=1,2,...Tpers
    epsilon    = scalar > 0, small value at which stitch f(X) function
    a          = scalar > 0, multiplicative factor in f(X) = a*exp(b*X)
    b          = scalar, multiplicative factor in exponent of
                 f(X) = a*exp(b*X)
    J          = integer >= 1, number of ability types
    S          = integer >= 3, number of periods in individual life
    Tpers      = integer > 3, number of time periods in the time path
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    omega_mat  = (S, J) matrix, omegas vector copied across J columns
    emat_arr   = (S, J, Tpers) array, ability matrix copied across Tpers
                 time periods
    lambda_arr = (S, J, Tpers) array, lambda_mat matrix copied across
                 Tpers time periods
    omega_arr  = (S, J, Tpers) array, omegas matrix copied across J
                 ability types
    L          = scalar > 0, aggregate labor
    L_cstr     = boolean or (T+S-1) boolean vector, =True if L < eps or
                 if L_t < eps

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L, L_cstr
    --------------------------------------------------------------------
    '''
    emat, lambdas, omegas = args
    epsilon = 0.01
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    J = len(lambdas)
    S = n_arr.shape[0]
    if n_arr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
        omega_mat = np.tile(omegas.reshape((S, 1)), (1, J))
        L = (lambda_mat * omega_mat * emat * n_arr).sum()
        L_cstr = L < epsilon
        if L_cstr:
            print('get_L() warning: distribution of labor supply ' +
                  'and/or parameters created L<epsilon')
            # Force L >= eps by stitching a * exp(b * L) for L < eps
            L = a * np.exp(b * L)
    elif n_arr.ndim == 3:  # This is the time path case
        Tpers = n_arr.shape[2]
        emat_arr = np.tile(emat.reshape((S, J, 1)), (1, 1, Tpers))
        lambda_arr = np.tile(lambdas.reshape((1, J, 1)), (S, 1, Tpers))
        omega_arr = np.tile(omegas.T.reshape((S, 1, Tpers)), (1, J, 1))
        L = ((lambda_arr * omega_arr * emat_arr * n_arr).sum(0)).sum(0)
        L_cstr = L < epsilon
        if L.min() < epsilon:
            print('Aggregate labor constraint is violated ' +
                  '(L_t < epsilon) for some period in time path.')
            L[L_cstr] = a * np.exp(b * L[L_cstr])

    return L, L_cstr


def get_K(b_arr, args):
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
    b_arr = (S, J) matrix or (S, J, Tpers) array, values for steady-
            state savings (b_{j,s}) or time path of the distribution of
            savings (b_{j,s,t})
    args  = length 4 tuple, (lambdas, omegas, imm_rates, g_n_arr)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    lambdas     = (J,) vector, income percentiles for ability types
    omegas      = (S,) vector or (S, Tpers) matrix, steady-state
                  population distribution (percents) by age or time path
                  of population distribution. Time path:
                  t=0,1,...Tpers-1
    imm_rates   = (S,) vector or (S, Tpers) matrix, long-run immigration
                  rates or time path of immigration rates. Time path:
                  t=1,2,...Tpers
    g_n_arr     = scalar or (Tpers,) vector, steady-state population
                  growth rate or time path of population growth rate.
                  Time path: t=1,2,...Tpers
    epsilon     = scalar > 0, small value at which stitch f(X) function
    a           = scalar > 0, multiplicative factor in f(X) = a*exp(b*X)
    b           = scalar, multiplicative factor in exponent of
                  f(X) = a*exp(b*X)
    J           = integer >= 1, number of ability types
    S           = integer >= 3, number of periods in individual life
    Tpers       = integer > 3, number of time periods in the time path
    lambda_mat  = (S, J) matrix, lambdas vector copied down S rows
    omega_mat   = (S, J) matrix, steady-state population distribution
                 copied across J columns
    omega_mat2  = (S, J) matrix, last S-1 elements of steady-state
                  population distribution with 0 appended as last
                  element, copied across J columns
    imm_rates2  = (S,) vector or (S, J, Tpers) matrix, last S-1 elements
                  of imm_rates with zero appended at end
    imm_mat     = (S, J) matrix, imm_rates2 copied across J columns
    lambda_arr  = (S, J, Tpers) array, lambda_mat matrix copied across
                  Tpers time periods
    omegas_arr  = (S, J, Tpers) array, omegas matrix copied across J
                  ability types
    omegas_arr2 = (S, J, Tpers) array, last S-1 elements of omega_arr
                  rows with zeros appended after each of them
    imm_arr     = (S, J, Tpers) array, immigration rates shifted forward
                  an age
    K           = scalar or (Tpers,) vector, steady-state aggregate
                  capital stock or time path of aggregate capital stock
    K_cstr      = boolean or (Tpers,) boolean vector, =True if K < eps
                  or if K_t < eps

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, K_cstr
    --------------------------------------------------------------------
    '''
    lambdas, omegas, imm_rates, g_n_arr = args
    epsilon = 0.01
    a = epsilon / np.exp(1)
    b = 1 / epsilon
    J = len(lambdas)
    S = b_arr.shape[0]
    if b_arr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
        omega_mat = np.tile(omegas.reshape((S, 1)), (1, J))
        omega_mat2 = np.tile(np.append(omegas[1:], 0).reshape((S, 1)),
                             (1, J))
        imm_rates2 = np.append(imm_rates[1:], 0)
        imm_mat = np.tile(imm_rates2.reshape((S, 1)), (1, J))
        K = (1 / (1 + g_n_arr)) * ((omega_mat + omega_mat2 * imm_mat) *
                                   lambda_mat * b_arr).sum()
        K_cstr = K < epsilon
        if K_cstr:
            print('get_K() warning: distribution of savings and/or ' +
                  'parameters created K<epsilon')
            # Force K >= eps by stitching a * exp(b *K) for K < eps
            K = a * np.exp(b * K)

    elif b_arr.ndim == 3:  # This is the time path case
        Tpers = b_arr.shape[2]
        lambda_arr = np.tile(np.reshape(lambdas, (1, J, 1)),
                             (S, 1, Tpers))
        omegas_arr = np.tile(omegas.T.reshape((S, 1, Tpers)),
                             (1, J, 1))
        omegas_arr2 = np.vstack((omegas_arr[1:, :, :],
                                 np.zeros((1, J, Tpers))))
        imm_arr = np.tile(imm_rates.T.reshape((S, 1, Tpers)), (1, J, 1))
        imm_arr2 = np.vstack((imm_arr[1:, :, :],
                              np.zeros((1, J, Tpers))))
        K = ((1 / (1 + g_n_arr)) *
             (((omegas_arr + omegas_arr2 * imm_arr2) * lambda_arr *
               b_arr).sum(0)).sum(0))
        K_cstr = K < epsilon
        if K.min() < epsilon:
            print('Aggregate capital constraint is violated ' +
                  '(K < epsilon) for some period in time path.')
            K[K_cstr] = a * np.exp(b * K[K_cstr])

    return K, K_cstr


def get_BQ(b_arr, r_arr, args):
    '''
    --------------------------------------------------------------------
    Solve for steady-state total bequests BQ or time path of total
    bequests BQ_t.
    --------------------------------------------------------------------
    INPUTS:
    b_arr = (S, J) matrix or (S, J, Tpers) array, values for steady-
            state savings (b_{j,s}) or time path of the distribution of
            savings (b_{j,s,t})
    r_arr = scalar > 0 or (Tpers,) vector, steady-state interest rate or
            time path of interest rates
    args  = length 4 tuple, (lambdas, omegas, mort_rates, g_n_arr)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    lambdas    = (J,) vector, income percentiles for ability types
    omegas     = (S,) vector or (S, Tpers) matrix, steady-state
                 population distribution (percents) by age or time path
                 of population distribution. Time path:
                 t=0,1,...Tpers-1
    mort_rates = (S,) vector, mortality rates by age
    g_n_arr    = scalar or (Tpers,) vector, steady-state population
                 growth rate or time path of population growth rate.
                 Time path: t=1,2,...Tpers
    J          = integer >= 1, number of ability types
    S          = integer >= 3, number of periods in individual life
    Tpers      = integer > 3, number of time periods in the time path
    rho_mat    = (S, J) matrix, mort_rates vector copied across J
                 columns
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    omega_mat  = (S, J) matrix, steady-state population distribution
                 copied across J columns
    rho_arr    = (S, J, Tpers) array, mort_rates copied across J types
                 and Tpers time periods
    lambda_arr = (S, J, Tpers) array, lambda_mat matrix copied across
                 Tpers time periods
    omegas_arr = (S, J, Tpers) array, omegas matrix copied across J
                 ability types
    BQ         = scalar or (Tpers,) vector, steady-state total bequests
                 or time path of total bequests

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: BQ
    --------------------------------------------------------------------
    '''
    lambdas, omegas, mort_rates, g_n_arr = args
    J = len(lambdas)
    S = b_arr.shape[0]
    if b_arr.ndim == 2:  # This is the steady-state case
        rho_mat = np.tile(mort_rates.reshape((S, 1)), (1, J))
        lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
        omega_mat = np.tile(omegas.reshape((S, 1)), (1, J))
        BQ = (((1 + r_arr) / (1 + g_n_arr)) *
              (rho_mat * lambda_mat * omega_mat * b_arr).sum())
    elif b_arr.ndim == 3:  # This is the time path case
        Tpers = b_arr.shape[2]
        rho_arr = np.tile(mort_rates.reshape((S, 1, 1)), (1, J, Tpers))
        lambda_arr = np.tile(lambdas.reshape((1, J, 1)), (S, 1, Tpers))
        omegas_arr = np.tile(omegas.T.reshape((S, 1, Tpers)), (1, J, 1))
        BQ = ((((1 + r_arr) / (1 + g_n_arr)) *
               (rho_arr * lambda_arr * omegas_arr *
                b_arr)).sum(0)).sum(0)

    return BQ


def get_I(b_arr, args):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate investment Inv or time path of
    aggregate investment I_t.
    --------------------------------------------------------------------
    INPUTS:
    b_arr = (S, J) matrix or (S, J, Tpers) array, values for steady-
            state savings (b_{j,s}) or time path of the distribution of
            savings (b_{j,s,t})
    args  = length 6 tuple,
            (lambdas, omegas, imm_rates, g_n_arr, g_y, delta)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_K()

    OBJECTS CREATED WITHIN FUNCTION:
    lambdas   = (J,) vector, income percentiles for ability types
    omegas    = (S,) vector or (S, Tpers) matrix, steady-state
                population distribution (percents) by age or time path
                of population distribution. Time path: t=0,1,...Tpers-1
    imm_rates = (S,) vector or (S, Tpers) matrix, long-run immigration
                rates or time path of immigration rates. Time path:
                t=1,2,...Tpers
    g_n_arr   = scalar or (Tpers,) vector, steady-state population
                growth rate or time path of population growth rate.
                Time path: t=1,2,...Tpers
    g_y       = scalar, labor productivity growth rate
    delta     = scalar in [0, 1], per period depreciation rate
    K_args    = length 4 tuple, arguments to pass into get_K() for
                steady-state K
    Tpers     = integer > 3, number of periods in time path
    Kt_args   = length 4 tuple, arguments to pass into get_K() for time
                path of K_t
    Ktp1_args = length 4 tuple, arguments to pass into get_K() for time
                path of K_{t+1}
    Inv       = scalar or (Tpers,) vector, steady-state aggregate
                investment or time path of aggregate investments

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Inv
    --------------------------------------------------------------------
    '''
    lambdas, omegas, imm_rates, g_n_arr, g_y, delta = args
    if b_arr.ndim == 2:  # This is the steady-state case
        K_args = (lambdas, omegas, imm_rates, g_n_arr)
        Inv = \
            ((np.exp(g_y) * (1 + g_n_arr) - 1 + delta) *
             get_K(b_arr, K_args)[0])
    elif b_arr.ndim == 3:  # This is the time path case
        Tpers = b_arr.shape[2]
        Inv = np.zeros(Tpers)
        Kt_args = (lambdas, omegas[:, :-1], imm_rates[:, :-1],
                   g_n_arr[:-1])
        Ktp1_args = (lambdas, omegas[:, 1:], imm_rates[:, 1:],
                     g_n_arr[1:])
        Inv[:-1] = (np.exp(g_y) * (1 + g_n_arr[1:]) *
                    get_K(b_arr[:, :, 1:], Ktp1_args)[0] -
                    (1 - delta) * get_K(b_arr[:, :, :-1], Kt_args)[0])
        Inv[-1] = Inv[-2]

    return Inv


def get_NX(b_arr, args):
    '''
    --------------------------------------------------------------------
    Solve for steady-state net exports NX or time path of net exports
    NX_t. In this model, net exports come exclusively from immigrants
    bringing capital into or out of the country
    --------------------------------------------------------------------
    INPUTS:
    b_arr = (S, J) matrix or (S, J, Tpers-1) array, values for steady-
            state savings (b_{j,s}) or time path of the distribution of
            savings (b_{j,s,t}) for t=2,3,...Tpers
    args  = length 4 tuple, (lambdas, omegas, imm_rates, g_y)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    lambdas    = (J,) vector, income percentiles for ability types
    omegas     = (S,) vector or (S, Tpers-1) matrix, steady-state
                 population distribution (percents) by age or time path
                 of population distribution. Time path:
                 t=1,2,...Tpers-1
    imm_rates  = (S,) vector or (S, Tpers-1) matrix, long-run
                 immigration rates or time path of immigration rates.
                 Time path: t=1,2,...Tpers-1
    g_y        = scalar, labor productivity growth rate
    J          = integer >= 1, number of ability types
    S          = integer >= 3, number of periods in individual life
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    omega_mat2 = (S, J) matrix, last S-1 elements of steady-state
                 population distribution with 0 appended as last
                 element, copied across J columns
    imm_rates2 = (S,) vector or (S, J, Tpers-1) matrix, last S-1
                 elements of imm_rates with zero appended at end
    imm_mat    = (S, J) matrix, imm_rates2 copied across J columns
    Tpers      = integer > 3, number of time periods in the time path
    lambda_arr = (S, J, Tpers-1) array, lambda_mat matrix copied across
                  Tpers-1 time periods
    omegas_arr  = (S, J, Tpers-1) array, omegas matrix copied across J
                  ability types
    omegas_arr2 = (S, J, Tpers-1) array, last S-1 elements of omega_arr
                  rows with zeros appended after each of them
    imm_arr     = (S, J, Tpers) array, immigration rates shifted forward
                  an age
    NX          = scalar or (Tpers,) vector, steady-state net exports
                  (coming exclusively from immigrants bringing capital
                  with them) or time path of net exports.
                  Time path: t= 2,3,...Tpers+1

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: NX
    --------------------------------------------------------------------
    '''
    lambdas, omegas, imm_rates, g_y = args
    J = len(lambdas)
    S = b_arr.shape[0]
    if b_arr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
        omega_mat2 = np.tile(np.append(omegas[1:], 0).reshape((S, 1)),
                             (1, J))
        imm_rates2 = np.append(imm_rates[1:], 0)
        imm_mat = np.tile(imm_rates2.reshape((S, 1)), (1, J))
        NX = -np.exp(g_y) * (omega_mat2 * imm_mat * lambda_mat *
                             b_arr).sum()
    elif b_arr.ndim == 3:  # This is the time path case
        Tpers = b_arr.shape[2]
        lambda_arr = np.tile(np.reshape(lambdas, (1, J, 1)),
                             (S, 1, Tpers))
        omegas_arr = np.tile(omegas.T.reshape((S, 1, Tpers)), (1, J, 1))
        omegas_arr2 = np.vstack((omegas_arr[1:, :, :],
                                 np.zeros((1, J, Tpers))))
        imm_arr = np.tile(imm_rates.T.reshape((S, 1, Tpers)), (1, J, 1))
        imm_arr2 = np.vstack((imm_arr[1:, :, :],
                              np.zeros((1, J, Tpers))))
        NX = -np.exp(g_y) * ((omegas_arr2 * imm_arr2 * lambda_arr *
                              b_arr).sum(0)).sum(0)

    return NX


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
             (Z, gamma)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Z     = scalar > 0, total factor productivity
    gamma = scalar in (0,1), capital share of income
    Y = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
        time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    Z, gamma = params
    Y = Z * (K ** gamma) * (L ** (1 - gamma))

    return Y


def get_C(c_arr, args):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    c_arr = (S, J) matrix or (S, J, Tpers) array, distribution of
            consumption c_{j,s} in steady state or time path for the
            distribution of consumption c_{j,s,t}
    args  = length 2 tuple, (lambdas, omegas)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    lambdas    = (J,) vector, income percentiles for distribution of
                 ability within each cohort
    omegas     = (S,) vector or (S, Tpers) matrix, steady-state
                 population distribution (percents) by age or time path
                 of population distribution. Time path: t=1,2,...Tpers
    J          = integer >= 1, number of ability types
    S          = integer >= 3, number of periods in individual life
    Tpers      = integer > 3, number of time periods in the time path
    lambda_mat = (S, J) matrix, lambdas vector copied down S rows
    omega_mat  = (S, J) matrix, omegas vector copied across J columns
    lambda_arr = (S, J, Tpers) array, lambda_mat matrix copied across
                 Tpers time periods
    C          = scalar > 0 or (Tpers,) vector, steady-state aggregate
                 consumption or time path of aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    lambdas, omegas = args
    J = len(lambdas)
    S = c_arr.shape[0]
    if c_arr.ndim == 2:  # This is the steady-state case
        lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
        omega_mat = np.tile(omegas.reshape((S, 1)), (1, J))
        C = (lambda_mat * omega_mat * c_arr).sum()
    elif c_arr.ndim == 3:  # This is the time path case
        Tpers = c_arr.shape[2]
        lambda_arr = np.tile(lambdas.reshape((1, J, 1)), (S, 1, Tpers))
        omega_arr = np.tile(omegas.T.reshape((S, 1, Tpers)), (1, J, 1))
        C = ((lambda_arr * omega_arr * c_arr).sum(0)).sum(0)

    return C
