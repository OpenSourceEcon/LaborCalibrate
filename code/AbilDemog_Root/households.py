'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with households' optimization in the steady-state or in the
transition path of the OG-ITA overlapping generations model with S-
period-lived agents, endogenous labor supply, and heterogeneous ability.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_cons()
    MU_c_stitch()
    MDU_n_stitch()
    get_n_errors()
    get_b_errors()
    get_bn_errors()
    get_cnb_mats()
    c1_bSp1err()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import aggregates as aggr
import firms

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def feasible(bmat, nmat, rpath, args):
    '''
    --------------------------------------------------------------------
    This function checks whether any constraints are violated by
    matrices for savings and labor supply given a time path for interest
    rates
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    '''
    (lambdas, emat, omega_vec, mort_rates, zeta_mat, g_y, l_tilde,
        g_n_SS, Z, gamma, delta) = args
    J = len(lambdas)
    b_feas = bmat > 0.0
    if (bmat <= 0.0).sum() > 0.0:
        err_msg = ('SS ERROR: Some value of initial guess for ' +
                   'bmat_ss is less than or equal to zero. ' +
                   'b_{j,s}<=0 for some j and s.')
        print('b_feas:', b_feas)
        raise RuntimeError(err_msg)
    n_feas = (nmat > 0.0) & (nmat < l_tilde)
    if ((nmat <= 0.0) | (nmat >= l_tilde)).sum() > 0.0:
        err_msg = ('SS ERROR: Some value of initial guess for ' +
                   'nmat_ss is less than or equal to zero or greater ' +
                   'than or equal to l_tilde. n_{j,s}<=0 or ' +
                   'n_{j,s}>=l_tildefor some j and s.')
        print('n_feas:', n_feas)
        raise RuntimeError(err_msg)
    w_params = (Z, gamma, delta)
    wpath = firms.get_w(rpath, w_params)
    b_s_mat = np.vstack((np.zeros(J), bmat[:-1, :]))
    b_sp1_mat = bmat
    BQ_args = (lambdas, omega_vec, mort_rates, g_n_SS)
    BQ = aggr.get_BQ(bmat, rpath, BQ_args)
    c_args = (emat, zeta_mat, lambdas, omega_vec, g_y)
    cmat = get_cons(rpath, wpath, b_s_mat, b_sp1_mat, nmat, BQ, c_args)
    c_feas = cmat > 0.0
    if (cmat <= 0.0).sum() > 0.0:
        err_msg = ('SS ERROR: Some value of initial guess for ' +
                   'bmat_ss or nmat_ss is not feasible because ' +
                   'resulting cmat_ss has a value that is less than ' +
                   'or equal to zero. c_{j,s}<=0 for some j and s.')
        print('c_feas:', c_feas)
        raise RuntimeError(err_msg)

    return b_feas, n_feas, c_feas


def get_cons(r, w, b, b_sp1, n, BQ, args):  # TODO: Finish docstring
    '''
    --------------------------------------------------------------------
    Calculate household consumption given prices, labor supply, current
    wealth, and savings
    --------------------------------------------------------------------
    INPUTS:
    r     = scalar or (per_rmn,) vector, current interest rate or time
            path of interest rates over remaining life periods
    w     = scalar or (per_rmn,) vector, current wage or time path of
            wages over remaining life periods
    e     = scalar or (per_rmn, J) matrix,
    b     = scalar or (per_rmn, J) matrix, current period wealth or time
            path of current period wealths over remaining life periods
    b_sp1 = scalar or (per_rmn, J) vector, savings for next period or
            time path of savings for next period over remaining life
            periods
    n     = scalar or (per_rmn, J) matrix, current labor supply or time
            path of labor supply over remaining life periods
    BQ    = scalar or (per_rmn,) vector, current total bequests or time
            path of total bequests over remaining life periods
    g_y   = scalar, labor productivity growth rate

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    cons = scalar or (p,) vector, current consumption or time path of
           consumption over remaining life periods

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cons
    --------------------------------------------------------------------
    '''
    e_js, zeta_js, lambdas, omega_path, g_y = args

    # These cases are the steady-state or the per_rmn=1 TPI
    if (b.ndim == 1) or (np.isscalar(b)):
        cons = ((1 + r) * b + w * e_js * n +
                zeta_js * (BQ / (lambdas * omega_path)) -
                np.exp(g_y) * b_sp1)

    # TPI cases for per_rmn > 1
    elif b.ndim == 2:
        J = len(lambdas)
        per_rmn = b.shape[0]
        lambdas_arr = np.tile(lambdas.reshape((1, J)), (per_rmn, 1))
        omega_arr = np.tile(omega_path.reshape((per_rmn, 1)), (1, J))
        r_arr = np.tile(r.reshape((per_rmn, 1)), (1, J))
        w_arr = np.tile(w.reshape((per_rmn, 1)), (1, J))
        BQ_arr = np.tile(BQ.reshape((per_rmn, 1)), (1, J))
        cons = (((1 + r_arr) * b) + (w_arr * e_js * n) +
                (zeta_js * (BQ_arr / (lambdas_arr * omega_arr))) -
                np.exp(g_y) * b_sp1)

    return cons


def MU_c_stitch(c_arr, sigma, graph=False):  # TODO: Finish docstring
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    cvec  = scalar or (p,) vector, individual consumption value or
            lifetime consumption over p consecutive periods
    sigma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-sigma) - 1) / (1 - sigma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 0.0001
    if np.ndim(c_arr) == 0:
        c_s = c_arr
        c_s_cnstr = c_s < epsilon
        if c_s_cnstr:
            b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
            b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
            MU_c = 2 * b2 * c_s + b1
        else:
            MU_c = c_s ** (-sigma)
    elif np.ndim(c_arr) == 1:
        p = c_arr.shape[0]
        cvec_cnstr = c_arr < epsilon
        MU_c = np.zeros(p)
        MU_c[~cvec_cnstr] = c_arr[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * c_arr[cvec_cnstr] + b1
    elif np.ndim(c_arr) == 2:  # This is when cvec is a 2D matrix
        p, J = c_arr.shape
        cvec_cnstr = c_arr < epsilon
        MU_c = np.zeros((p, J))
        MU_c[~cvec_cnstr] = c_arr[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * c_arr[cvec_cnstr] + b1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        cvec_CRRA   = (1000,) vector, support of c including values
                      between 0 and epsilon
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.00005, epsilon, 500)
        MU_stitch = 2 * b2 * cvec_stitch + b1
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.00005, epsilon * 3))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MU_c


def MDU_n_stitch(n_arr, params, graph=False):  # TODO: Finish docstring
    '''
    --------------------------------------------------------------------
    Generate marginal disutility(ies) of labor with elliptical
    disutility of labor function and stitched functions at lower bound
    and upper bound of labor supply such that the new hybrid function is
    defined over all labor supply on the real line but the function has
    similar properties to the Inada conditions at the upper and lower
    bounds.

    v'(n) = (b / l_tilde) * ((n / l_tilde) ** (upsilon - 1)) *
            ((1 - ((n / l_tilde) ** upsilon)) ** ((1-upsilon)/upsilon))
            if n >= eps_low <= n <= eps_high
          = g_low'(n)  = 2 * b2 * n + b1 if n < eps_low
          = g_high'(n) = 2 * d2 * n + d1 if n > eps_high

        such that g_low'(eps_low) = u'(eps_low)
        and g_low''(eps_low) = u''(eps_low)
        and g_high'(eps_high) = u'(eps_high)
        and g_high''(eps_high) = u''(eps_high)

        v(n) = -b *(1 - ((n/l_tilde) ** upsilon)) ** (1/upsilon)
        g_low(n)  = b2 * (n ** 2) + b1 * n + b0
        g_high(n) = d2 * (n ** 2) + d1 * n + d0
    --------------------------------------------------------------------
    INPUTS:
    nvec   = scalar or (p,) vector, labor supply value or labor supply
             values over remaining periods of lifetime
    params = length 3 tuple, (l_tilde, b_ellip, upsilon)
    graph  = Boolean, =True if want plot of stitched marginal disutility
             of labor function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    l_tilde       = scalar > 0, time endowment for each agent each per
    b_ellip       = scalar > 0, scale parameter for elliptical utility
                    of leisure function
    upsilon       = scalar > 1, shape parameter for elliptical utility
                    of leisure function
    eps_low       = scalar > 0, positive value close to zero
    eps_high      = scalar > 0, positive value just less than l_tilde
    n_s           = scalar, individual labor supply
    n_s_low       = boolean, =True for n_s < eps_low
    n_s_high      = boolean, =True for n_s > eps_high
    n_s_uncstr    = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high
    MDU_n         = scalar or (p,) vector, marginal disutility or
                    marginal utilities of labor supply
    b1            = scalar, intercept value in linear marginal
                    disutility of labor at lower bound
    b2            = scalar, slope coefficient in linear marginal
                    disutility of labor at lower bound
    d1            = scalar, intercept value in linear marginal
                    disutility of labor at upper bound
    d2            = scalar, slope coefficient in linear marginal
                    disutility of labor at upper bound
    p             = integer >= 1, number of periods remaining in life
    nvec_s_low    = boolean, =True for n_s < eps_low
    nvec_s_high   = boolean, =True for n_s > eps_high
    nvec_s_uncstr = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high

    FILES CREATED BY THIS FUNCTION:
        MDU_n_stitched.png

    RETURNS: MDU_n
    --------------------------------------------------------------------
    '''
    l_tilde, b_ellip, upsilon = params
    eps_low = 0.000001
    eps_high = l_tilde - 0.000001
    # This if is for when nvec is a scalar
    if np.ndim(n_arr) == 0:
        n_s = n_arr
        n_s_low = n_s < eps_low
        n_s_high = n_s > eps_high
        n_s_uncstr = (n_s >= eps_low) and (n_s <= eps_high)
        if n_s_uncstr:
            MDU_n = ((b_ellip / l_tilde) *
                     ((n_s / l_tilde) ** (upsilon - 1)) *
                     ((1 - ((n_s / l_tilde) ** upsilon)) **
                      ((1 - upsilon) / upsilon)))
        elif n_s_low:
            b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) *
                  (upsilon - 1) * (eps_low ** (upsilon - 2)) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) **
                   ((1 - upsilon) / upsilon)) *
                  (1 + ((eps_low / l_tilde) ** upsilon) *
                   ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
            b1 = ((b_ellip / l_tilde) *
                  ((eps_low / l_tilde) ** (upsilon - 1)) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) **
                   ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
            MDU_n = 2 * b2 * n_s + b1
        elif n_s_high:
            d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) *
                  (upsilon - 1) * (eps_high ** (upsilon - 2)) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) **
                   ((1 - upsilon) / upsilon)) *
                  (1 + ((eps_high / l_tilde) ** upsilon) *
                   ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
            d1 = ((b_ellip / l_tilde) *
                  ((eps_high / l_tilde) ** (upsilon - 1)) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) **
                   ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
            MDU_n = 2 * d2 * n_s + d1
    # This if is for when nvec is a one-dimensional vector
    elif np.ndim(n_arr) == 1:
        p = n_arr.shape[0]
        nvec_low = n_arr < eps_low
        nvec_high = n_arr > eps_high
        nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
        MDU_n = np.zeros(p)
        MDU_n[nvec_uncstr] = (
            (b_ellip / l_tilde) *
            ((n_arr[nvec_uncstr] / l_tilde) ** (upsilon - 1)) *
            ((1 - ((n_arr[nvec_uncstr] / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_low ** (upsilon - 2)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) *
              (1 + ((eps_low / l_tilde) ** upsilon) *
               ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
        b1 = ((b_ellip / l_tilde) *
              ((eps_low / l_tilde) ** (upsilon - 1)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
        MDU_n[nvec_low] = 2 * b2 * n_arr[nvec_low] + b1
        d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_high ** (upsilon - 2)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) *
              (1 + ((eps_high / l_tilde) ** upsilon) *
               ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
        d1 = ((b_ellip / l_tilde) *
              ((eps_high / l_tilde) ** (upsilon - 1)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
        MDU_n[nvec_high] = 2 * d2 * n_arr[nvec_high] + d1
    # This if is for when n_arr is a 2D matrix
    elif np.ndim(n_arr) == 2:
        p, J = n_arr.shape
        nvec_low = n_arr < eps_low
        nvec_high = n_arr > eps_high
        nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
        MDU_n = np.zeros((p, J))
        MDU_n[nvec_uncstr] = (
            (b_ellip / l_tilde) *
            ((n_arr[nvec_uncstr] / l_tilde) ** (upsilon - 1)) *
            ((1 - ((n_arr[nvec_uncstr] / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_low ** (upsilon - 2)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) *
              (1 + ((eps_low / l_tilde) ** upsilon) *
               ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
        b1 = ((b_ellip / l_tilde) *
              ((eps_low / l_tilde) ** (upsilon - 1)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
        MDU_n[nvec_low] = 2 * b2 * n_arr[nvec_low] + b1
        d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_high ** (upsilon - 2)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) *
              (1 + ((eps_high / l_tilde) ** upsilon) *
               ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
        d1 = ((b_ellip / l_tilde) *
              ((eps_high / l_tilde) ** (upsilon - 1)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
               ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
        MDU_n[nvec_high] = 2 * d2 * n_arr[nvec_high] + d1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        nvec_ellip  = (1000,) vector, support of n including values
                      between 0 and eps_low and between eps_high and
                      l_tilde
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        nvec_ellip = np.linspace(eps_low / 2, eps_high +
                                 ((l_tilde - eps_high) / 5), 1000)
        MDU_ellip = (
            (b_ellip / l_tilde) *
            ((nvec_ellip / l_tilde) ** (upsilon - 1)) *
            ((1 - ((nvec_ellip / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        n_stitch_low = np.linspace(-0.05, eps_low, 500)
        MDU_stitch_low = 2 * b2 * n_stitch_low + b1
        n_stitch_high = np.linspace(eps_high, l_tilde + 0.000005, 500)
        MDU_stitch_high = 2 * d2 * n_stitch_high + d1
        fig, ax = plt.subplots()
        plt.plot(nvec_ellip, MDU_ellip, ls='solid', color='black',
                 label='$v\'(n)$: Elliptical')
        plt.plot(n_stitch_low, MDU_stitch_low, ls='dashed', color='red',
                 label='$g\'(n)$: low stitched')
        plt.plot(n_stitch_high, MDU_stitch_high, ls='dotted',
                 color='blue', label='$g\'(n)$: high stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Labor $n$')
        plt.ylabel(r'Marginal disutility $v\'(n)$')
        plt.xlim((-0.05, l_tilde + 0.01))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "MDU_n_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MDU_n


def get_n_errors(n_vec, c_vec, wpath, args):  # TODO: Finish docstring
    '''
    --------------------------------------------------------------------
    Generates vector of static Euler errors that characterize the
    optimal lifetime labor supply decision. Because this function is
    used for solving for lifetime decisions in both the steady-state and
    in the transition path, lifetimes will be of varying length.
    Lifetimes in the steady-state will be S periods. Lifetimes in the
    transition path will be p in [2, S] periods
    --------------------------------------------------------------------
    INPUTS:
    n_arr = (p,) vector, distribution of labor supply by age n_p
    args  = either length 8 tuple or length 10 tuple, is (w, sigma,
            l_tilde, chi_n_vec, b_ellip, upsilon, diff, cvec) in most
            cases. In the time path for the age-S individuals in period
            1, the tuple is (wpath, sigma, l_tilde, chi_n_vec, b_ellip,
            upsilon, diff, rpath, b_s_vec, b_sp1_vec)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()
        MDU_n_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    w           = scalar > 0 or (p,) vector, steady-state wage or time
                  path of wage
    sigma       = scalar > 0, coefficient of relative risk aversion
    l_tilde     = scalar > 0, time endowment of each agent in each per
    chi_n_vec   = (p,) vector, values for chi^n_p
    b_ellip     = scalar > 0, fitted value of b for elliptical
                  disutility of labor
    upsilon     = scalar > 1, fitted value of upsilon for elliptical
                  disutility of labor
    diff        = Boolean, =True if use simple difference Euler errors.
                  Use percent difference errors otherwise
    cvec        = (p,) vector, distribution of consumption by age c_p
    mu_c        = (p,) vector, marginal utility of consumption
    mdun_params = length 3 tuple, (l_tilde, b_ellip, upsilon)
    mdu_n       = (p,) vector, marginal disutility of labor supply
    n_errors    = (p,) vector, Euler errors characterizing optimal labor
                  supply nvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: n_errors
    --------------------------------------------------------------------
    '''
    evec, chi_n_vec, sigma, l_tilde, b_ellip, upsilon = args
    mu_c = MU_c_stitch(c_vec, sigma)
    mdun_params = (l_tilde, b_ellip, upsilon)
    mdu_n = MDU_n_stitch(n_vec, mdun_params)
    n_errors = wpath * evec * mu_c - chi_n_vec * mdu_n

    return n_errors


def get_b_errors(c_vec, b_vec, rpath, args):  # TODO: Finish docstring
    '''
    --------------------------------------------------------------------
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings decision. Because this function is used for
    solving for lifetime decisions in both the steady-state and in the
    transition path, lifetimes will be of varying length. Lifetimes in
    the steady-state will be S periods. Lifetimes in the transition path
    will be p in [2, S] periods
    --------------------------------------------------------------------
    INPUTS:
    cmat = (per_rmn,) vector or (per_rmn, J) matrix, distribution of consumption
    bmat = (per_rmn,) vector or (per_rmn, J) matrix, distribution of savings
    r    = scalar > 0 or (per_rmn-1,) vector, steady-state interest rate or
           time path of interest rates
    args = length 5 tuple, (beta, sigma, chi_b_vec, mort_rates, g_y)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        MU_c_stitch()

    OBJECTS CREATED WITHIN FUNCTION:
    beta     = scalar in (0,1), discount factor
    sigma    = scalar > 0, coefficient of relative risk aversion
    diff     = boolean, =True if use simple difference Euler errors. Use
               percent difference errors otherwise.
    mu_c     = (p-1,) vector, marginal utility of current consumption
    mu_cp1   = (p-1,) vector, marginal utility of next period consumpt'n
    b_errors = (p-1,) vector, Euler errors characterizing optimal
               savings bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    beta, sigma, chi_b_vec, mort_rates, g_y = args
    if c_vec.shape[0] == 1:
        mu_c = MU_c_stitch(c_vec[0], sigma)
        mu_bp1 = MU_c_stitch(b_vec[0], sigma)
        b_errors = np.exp(-sigma * g_y) * chi_b_vec * mu_bp1 - mu_c
    elif c_vec.shape[0] > 1:
        b_errors = np.zeros_like(b_vec)
        mu_c = MU_c_stitch(c_vec[:-1], sigma)
        mu_cp1 = MU_c_stitch(c_vec[1:], sigma)
        mu_bp1 = MU_c_stitch(b_vec, sigma)
        b_errors[:-1] = \
            ((np.exp(-sigma * g_y) *
              (chi_b_vec * mort_rates[:-1] * mu_bp1[:-1] +
               beta * (1 - mort_rates[:-1]) * (1 + rpath) * mu_cp1)) -
             mu_c)
        b_errors[-1] = (np.exp(-sigma * g_y) * chi_b_vec * mu_bp1[-1] -
                        mu_cp1[-1])

    return b_errors


def bn_errors(bn_vec, *args):
    '''
    --------------------------------------------------------------------
    Computes labor supply and savings Euler errors for given b_{s+1} and
    n_s vectors and given the path of interest rates and wages
    --------------------------------------------------------------------
    INPUTS:
    bn_vec = (2p-1,) vector, values for remaining life n_s and b_{s+1}
    args   = length 11 tuple, (rpath, wpath, b_init, p, beta, sigma,
             l_tilde, chi_n_vec, b_ellip, upsilon, diff)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_cons()
        get_n_errors()
        get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    rpath     = (p,) vector, time path of interest rates over remaining
                lifetime
    wpath     = (p,) vector, time path of wages over remaining lifetime
    b_init    = scalar, initial wealth
    p         = integer >= 1, number of periods remaining in lifetime
    beta      = scalar in (0,1), discount factor
    sigma     = scalar > 0, coefficient of relative risk aversion
    l_tilde   = scalar > 0, per-period time endowment for every agent
    chi_n_vec = (p,) vector, values for chi^n_s for remaining lifetime
    b_ellip   = scalar > 0, fitted value of b for elliptical disutility
                of labor
    upsilon   = scalar > 1, fitted value of upsilon for elliptical
                disutility of labor
    b_sp1_vec = (p-1,) vector, savings over remaining lifetime
    n_vec     = (p,) vector, labor supply over remaining lifetime
    b_s_vec   = (p-1,) vector, wealth over remaining lifetime
    n_args    = length 8 tuple, args to pass into hh.get_n_errors()
    n_errors  = (p,) vector, remaining lifetime labor supply Euler
                errors
    b_args    = length 3 tuple, args to pass into hh.get_b_errors()
    b_errors  = (p-1,) vector, remaining lifetime savings Euler errors
    bn_errors = (2p-1,) vector, remaining lifetime savings and labor
                supply Euler errors

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bn_errors
    --------------------------------------------------------------------
    '''
    (rpath, wpath, BQpath, b_init, lambdas, evec, zeta_vec, omega_vec,
        mort_rates, per_rmn, beta, sigma, l_tilde, chi_n_vec, chi_b_vec,
        b_ellip, upsilon, g_y) = args
    b_sp1 = bn_vec[:per_rmn]
    b_s = np.append(b_init, bn_vec[:per_rmn - 1])
    n_vec = bn_vec[per_rmn:]
    c_args = (evec, zeta_vec, lambdas, omega_vec, g_y)
    c_vec = get_cons(rpath, wpath, b_s, b_sp1, n_vec, BQpath, c_args)
    n_args = (evec, chi_n_vec, sigma, l_tilde, b_ellip, upsilon)
    n_errors = get_n_errors(n_vec, c_vec, wpath, n_args)
    b_args = (beta, sigma, chi_b_vec, mort_rates, g_y)
    if per_rmn == 1:
        b_errors = get_b_errors(c_vec, b_sp1, 0, b_args)
    else:
        b_errors = get_b_errors(c_vec, b_sp1, rpath[1:], b_args)
    bn_errors = np.hstack((b_errors, n_errors))

    return bn_errors


def get_bsp1_stitch(varbls, params):
    '''
    --------------------------------------------------------------------
    Generates vector of steady-state savings \bar{b}_{j,s+1} or
    transition path savings b_{j,s+1,t+t} for all type-J households of
    age s given interest rate, wage, current wealth, ability, labor
    supply, consumption, and total bequests.  We have to stitch a
    function inside this function the bequest motive requires that
    savings be strictly positive.

    Let x = b_{j,s+1,t+1} and y = adjusted b_{j,s+1,t+1} > 0
    Let g(x) = e^{a * x + b} be the stitched function such that
    For x >= epsilon, let y = b_{j,s+1,t+1}
    For x < epsilon, let y = e^{aa * b_{j,s+1,t+1} + bb}
    where g(eps) = eps
    and g'(eps) = 1
    which implies a = 1/eps, and b = ln(eps) - 1
    --------------------------------------------------------------------
    INPUTS:
    varbls = length 6 tuple, variable arguments in budget constraint
             (r, w, BQ, bvec, cvec, nvec)
    params = length 5 tuple, parameter arguments in budget constraint
             (evec, zeta_vec, lambdas, omega_s, g_y)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    r         = scalar > 0, steady-state or period-t interest rate
    w         = scalar > 0, steady-state or period-t wage
    bvec      = (J,) vector, current period wealth of all J lifetime
                income group households of age-s
    cvec      = (J,) vector, current period consumption of all J
                lifetime income group households of age-s
    nvec      = (J,) vector, current period labor supply of all J
                lifetime income group households of age-s
    BQ        = scalar > 0, total bequests
    evec      = (J,) vector, age-s distribution of labor productivity
    g_y       = scalar, labor productivity growth rate
    zeta_vec  = (J,) vector, distribution of total bequests to the J
                lifetime income group households of age s
    lambdas   = (J,) vector, income percentiles for distribution of
                ability within each cohort
    omega_s   = scalar >= 0, percent of population that is age s
    b_sp1_vec = (J,) vector, savings implied by budget constraint of J
                lifetime income group households of age s

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_sp1_vec
    --------------------------------------------------------------------
    '''
    r, w, BQ, bvec, cvec, nvec = varbls
    evec, zeta_vec, lambdas, omega_s, g_y = params
    epsilon = 0.005
    aa = 1 / epsilon
    bb = np.log(epsilon) - 1
    b_sp1_vec = np.exp(-g_y) * (((1 + r) * bvec) + (w * evec * nvec) +
                                (zeta_vec * BQ / (omega_s * lambdas)) -
                                cvec)
    b_cstr = b_sp1_vec < epsilon
    if b_cstr.sum() > 0:
        b_sp1_adj = np.zeros_like(b_sp1_vec)
        b_sp1_vec[~b_cstr] = b_sp1_vec[~b_cstr]
        b_sp1_adj[b_cstr] = np.exp(aa * b_sp1_vec[b_cstr] +
                                   bb * np.ones_like(b_sp1_vec[b_cstr]))
    else:
        b_sp1_adj = b_sp1_vec

    return b_sp1_adj


def get_csp1_stitch(varbls, params):
    '''
    --------------------------------------------------------------------
    Generates vector of steady-state consumption \bar{c}_{j,s+1} or
    transition path consumption c_{j,s+1,t+t} for all type-J households
    of age s given current consumption, current savings, and the
    interest rate. We have to stitch a function inside this function
    because the numerator cannot be negative.

    Let x = chi^b_j * rho_s * (b_{j,s+1,t+1} ** (-sigma))
    Let y = e^{sigma * g_y} * (c_{j,s,t} ** (-sigma))
    Let g(x) = y - e^{-(a * x + b)} be the stitched function such that
    ub < y
    for x > ub - eps, x = g(x)
    where g(ub - eps) = ub - eps
    and g'(ub - eps) = 1
    which implies a = 1/eps, and b = -ln(eps) - (1/eps) * (ub - eps)
    --------------------------------------------------------------------
    INPUTS:
    varbls = length 3 tuple, variable arguments in Euler equation
             (r, cvec, bsp1_vec)
    params = length 5 tuple, parameter arguments in Euler equation
             (chi_b_vec, beta, sigma, rho_s, g_y)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    r         = scalar > 0, steady-state or period-(t+1) interest rate
    bsp1_vec  = (J,) vector, current period savings of all J lifetime
                income group households of age-s
    cvec      = (J,) vector, current period consumption of all J
                lifetime income group households of age-s
    g_y       = scalar, labor productivity growth rate
    sigma     = scalar > 0, coefficient of relative risk aversion
    beta      = scalar in (0, 1), per period discount factor
    chi_b_vec = (J,) vector, chi^b_j value for all J income groups
    rho_s     = scalar in [0, 1], mortality rate of age-s household
    csp1_vec  = (J,) vector, next period consumption implied by savings
                Euler equation of J lifetime income group households of
                age s

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: csp1_vec
    --------------------------------------------------------------------
    '''
    r, cvec, bsp1_vec = varbls
    chi_b_vec, beta, sigma, rho_s, g_y = params
    epsilon = 0.001
    x_vec = chi_b_vec * rho_s * MU_c_stitch(bsp1_vec, sigma)
    y_vec = np.exp(sigma * g_y) * MU_c_stitch(cvec, sigma)
    ub_vec = y_vec - 0.001
    aa = 1 / epsilon
    bb = -np.log(epsilon) - aa * (ub_vec - epsilon)
    x_cstr = x_vec > ub_vec - epsilon
    csp1_vec = np.zeros_like(cvec)
    csp1_vec[~x_cstr] = ((y_vec[~x_cstr] - x_vec[~x_cstr]) /
                         (beta * (1 - rho_s) * (1 + r))) ** (-1 / sigma)
    if x_cstr.sum() > 0:
        x_vec_adj = (ub_vec[x_cstr] - np.exp(-(aa * x_vec[x_cstr] +
                                               bb[x_cstr])))
        csp1_vec[x_cstr] = (((y_vec[x_cstr] - x_vec_adj) /
                             (beta * (1 - rho_s) * (1 + r))) **
                            (-1 / sigma))

    return csp1_vec


# def get_cnb_mats(c_init, b_init, rpath, wpath, BQpath, args):
#     '''
#     --------------------------------------------------------------------
#     Generate lifetime consumption vector for all J ability type
#     households with per_rmn periods remaining in their lifetimes given a
#     guess for initial consumption c_{j,S-per_rmn+1}, initial wealth
#     b_{j,S-p+1}, a time path for interest rates and wages, and
#     parameters using household Euler conditions and budget constraints.
#     --------------------------------------------------------------------
#     INPUTS:
#     c_init = (J,), consumption in initial period of lifetime
#              c_{j,S-per_rmn+1}
#     rpath  = (per_rmn,) vector, path of interest rates over lifetime
#     wpath  = (per_rmn,) vector, path of wages over lifetime
#     args   = length 9 tuple, (b_init, emat, beta, sigma, l_tilde,
#              b_ellip, upsilon, chi_n_vec, SS_tol)

#     OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
#         get_bsp1()
#         get_csp1_stitch()
#         get_n_errors()
#         get_b_errors()

#     OBJECTS CREATED WITHIN FUNCTION:
#     b_init    = (J,) vector, initial wealth b_{j,S-per_rmn+1} for all J
#                 ability types
#     emat      = (per_rmn, J) matrix, remaining lifetime ability profiles
#                 for all J ability types
#     beta      = scalar in (0, 1), discount factor
#     sigma     = scalar >= 1, coefficient of relative risk aversion
#     l_tilde   = scalar > 0, per-period time endowment for every agent
#     b_ellip   = scalar > 0, fitted value of b for elliptical disutility
#                 of labor
#     upsilon   = scalar > 1, fitted value of upsilon for elliptical
#                 disutility of labor
#     chi_n_vec = (per_rmn,) vector, values for chi^n_s over remaining
#                 lifetime
#     SS_tol    = scalar > 0, tolerance for inner-loop root finder
#     J         = integer >= 1, number of heterogeneous ability groups
#     per_rmn   = integer >= 1, number of periods remaining in a
#                 household's life
#     cmat      = (per_rmn, J) matrix, household remaining lifetime
#                 consumption for all J ability types given c_j1 and b_j1
#     nmat      = (per_rmn, J) matrix, household remaining lifetime labor
#                 supply for all J ability types given c_j1 and b_j1
#     bmat      = (per_rmn+1, J) matrix, household remaining lifetime
#                 savings for all J ability types given c_j1 and b_j1
#     n_err_mat = (per_rmn, J) matrix, household remaining lifetime labor
#                 supply Euler errors for all J ability types
#     b_err_mat = (per_rmn+1, J) matrix, household remaining lifetime
#                 savings Euler errors for all J ability types
#     per       = integer >= 1, index of period number
#     nvec_init = (J,) vector, initial guess for n_{j,s} given c_{j,s} for
#                 all J ability types
#     n_args    = length 8 tuple, arguments to pass into get_n_errors()
#     result_n  = results object, solution from opt.root(get_n_errors,...)
#     b_args    = length 2 tuple, (beta, sigma)

#     FILES CREATED BY THIS FUNCTION: None

#     RETURNS: cmat, nmat, bmat, n_err_mat, b_err_mat
#     --------------------------------------------------------------------
#     '''
#     (J, lambdas, emat, mort_rates, omega_vec, zeta_mat, chi_n_vec,
#         chi_b_vec, beta, sigma, l_tilde, b_ellip, upsilon, g_y,
#         SS_tol) = args
#     per_rmn = rpath.shape[0]
#     cmat = np.zeros((per_rmn, J))
#     nmat = np.zeros((per_rmn, J))
#     bmat = np.zeros((per_rmn + 1, J))
#     n_err_mat = np.zeros((per_rmn, J))

#     for per in range(per_rmn):
#         if per == 0:
#             bmat[per, :] = b_init
#             cmat[per, :] = c_init
#             nvec_init = (l_tilde / 2) * np.ones(J)
#         else:
#             b_varbls = (rpath[per - 1], wpath[per - 1], BQpath[per - 1],
#                         bmat[per - 1, :], cmat[per - 1, :],
#                         nmat[per - 1, :])
#             b_params = (emat[per - 1, :], zeta_mat[per - 1, :], lambdas,
#                         omega_vec[per - 1], g_y)
#             bmat[per, :] = get_bsp1_stitch(b_varbls, b_params)
#             c_varbls = (rpath[per], cmat[per - 1, :], bmat[per, :])
#             c_params = (chi_b_vec, beta, sigma, mort_rates[per - 1],
#                         g_y)
#             cmat[per, :] = get_csp1_stitch(c_varbls, c_params)
#             nvec_init = nmat[per - 1, :]
#         n_args = (wpath[per], emat[per, :], sigma, l_tilde,
#                   chi_n_vec[per], b_ellip, upsilon, cmat[per, :])
#         result_n = \
#             opt.root(get_n_errors, nvec_init, args=(n_args),
#                      method='lm', tol=SS_tol)
#         nmat[per, :] = result_n.x
#         n_err_mat[per, :] = result_n.fun
#     b_varbls = (rpath[-1], wpath[-1], BQpath[-1], bmat[-2, :],
#                 cmat[-1, :], nmat[-1, :])
#     b_params = (emat[-1, :], zeta_mat[-1, :], lambdas, omega_vec[-1],
#                 g_y)
#     bmat[-1, :] = get_bsp1_stitch(b_varbls, b_params)

#     berr_args = (beta, sigma, chi_b_vec, mort_rates, g_y)
#     b_err_mat = get_b_errors(cmat, bmat[1:, :], rpath[1:], berr_args)

#     return cmat, nmat, bmat, n_err_mat, b_err_mat


def get_cnb_mats(bmat_init, nmat_init, b_init, rpath, wpath, BQpath,
                 args):
    '''
    --------------------------------------------------------------------
    Given time paths for interest rates and wages, this function
    generates matrices for the time path of the distribution of
    individual consumption, labor supply, savings, the corresponding
    Euler errors for the labor supply decision and the savings decision,
    and the residual error of end-of-life savings associated with
    solving each lifetime decision.
    --------------------------------------------------------------------
    INPUTS:
    rpath = (T2+S-1,) vector, equilibrium time path of interest rate
    wpath = (T2+S-1,) vector, equilibrium time path of the real wage
    args  = length 13 tuple, (J, S, T2, emat, beta, sigma, l_tilde,
            b_ellip, upsilon, chi_n_vec, bmat1, n_ss, In_Tol)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.c1_bSp1err()
        hh.get_cnb_vecs()
        hh.get_n_errors()
        hh.get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    J             =
    S             = integer in [3,80], number of periods an individual
                    lives
    T2            = integer > S, number of periods until steady state
    emat          = (S, J) matrix
    beta          = scalar in (0,1), discount factor
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    bmat1         = (S, J) matrix, initial period household savings
                    distribution
    In_Tol        = scalar > 0, tolerance level for TPI inner-loop root
                    finders
    cpath         = (S, J, T2+S-1) array, time path of the distribution
                    of household consumption
    npath         = (S, J, T2+S-1) array, time path of the distribution
                    of household labor supply
    bpath         = (S, J, T2+S-1) array, time path of the distribution
                    of household savings
    n_err_path    = (S, J, T2+S-1) matrix, time path of household labor
                    supply Euler errors
    b_err_path    = (S, J, T2+S-1) matrix, time path of household
                    savings Euler errors
    per_rmn       = integer in [1, S-1], index representing number of
                    periods remaining in a lifetime, used to solve
                    incomplete lifetimes
    j_ind         = integer in [0, J-1], index representing ability type
    n_S1_init     = scalar in (0, l_tilde), initial guess for n_{j,S,1}
    nS1_args      = length 10 tuple, args to pass in to
                    hh.get_n_errors()
    results_nS1   = results object, results from
                    opt.root(hh.get_n_errors,...)
    DiagMaskb     = (per_rmn-1, per_rmn-1) boolean identity matrix
    DiagMaskn     = (per_rmn, per_rmn) boolean identity matrix
    b_sp1_init    = (per_rmn-1,) vector, initial guess for remaining
                    lifetime savings for household j in period t
    n_s_init      = (per_rmn,) vector, initial guess for remaining
                    lifetime labor supply for household j in period t
    bn_init       = (2*per_rmn - 1,) vector, initial guess for remaining
                    lifetime savings and labor supply for household j in
                    period t
    bn_args       = length 11 tuple, arguments to pass into
                    opt.root(hh.bn_errors,...)
    results_bn    = results object, results from opt.root(hh.bn_errors,)
    bvec          = (per_rmn-1,) vector, optimal savings given rpath and
                    wpath
    nvec          = (per_rmn,) vector, optimal labor supply given rpath
                    and wpath
    b_errors      = (per_rmn-1,) vector, savings Euler errors
    n_errors      = (per_rmn,) vector, labor supply Euler errors
    b_s_vec       = (per_rmn,) vector, remaining lifetime beginning of
                    period wealth for household j in period t
    b_sp1_vec     = (per_rmn,) vector, remaining lifetime savings for
                    household j in period t
    cvec          = (per_rmn,) vector, remaining lifetime consumption
                    for household j in period t
    t             = integer in [0, T2-1], index of time period

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cpath, npath, bpath, n_err_path, b_err_path, bSp1_err_path
    --------------------------------------------------------------------
    '''
    (per_rmn, lambdas, emat, mort_rates, omega_vec, zeta_mat, chi_n_vec,
        chi_b_vec, beta, sigma, l_tilde, b_ellip, upsilon, g_y,
        g_n_path, SS_tol) = args
    J = len(lambdas)
    nmat = np.zeros((per_rmn, J))
    bmat = np.zeros((per_rmn, J))
    n_err_mat = np.zeros((per_rmn, J))
    b_err_mat = np.zeros((per_rmn, J))

    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    for j_ind in range(J):
        bn_args = (rpath, wpath, BQpath, b_init[j_ind], lambdas[j_ind],
                   emat[:, j_ind], zeta_mat[:, j_ind], omega_vec,
                   mort_rates, per_rmn, beta, sigma, l_tilde, chi_n_vec,
                   chi_b_vec[j_ind], b_ellip, upsilon, g_y)
        bn_init = np.append(bmat_init[:, j_ind].flatten(),
                            nmat_init[:, j_ind].flatten())
        results_bn = opt.root(bn_errors, bn_init, args=(bn_args),
                              method='lm', tol=SS_tol)
        bmat[:, j_ind] = results_bn.x[:per_rmn]
        nmat[:, j_ind] = results_bn.x[per_rmn:]
        b_err_mat[:, j_ind] = results_bn.fun[:per_rmn]
        n_err_mat[:, j_ind] = results_bn.fun[per_rmn:]
    b_s_mat = np.vstack((np.zeros(J), bmat[:-1, :]))
    b_sp1_mat = bmat
    c_args = (emat, zeta_mat, lambdas, omega_vec, g_y)
    cmat = get_cons(rpath, wpath, b_s_mat, b_sp1_mat, nmat, BQpath,
                    c_args)

    return cmat, nmat, bmat, n_err_mat, b_err_mat
