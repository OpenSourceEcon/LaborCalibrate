'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and endogenous labor supply from
Chapter 7 of the OG textbook.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
    inner_loop()
    get_SS()
    create_graphs()
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import households as hh
import firms
import aggregates as aggr
import utilities as utils
import consumption as consump
import hrs_by_age as hrs

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def inner_loop(r, w, args):
    '''
    --------------------------------------------------------------------
    Given values for r and w, solve for equilibrium errors from the two
    first order conditions of the firm
    --------------------------------------------------------------------
    INPUTS:
    r    = scalar > 0, guess at steady-state interest rate
    w    = scalar > 0, guess at steady-state wage
    args = length 14 tuple, (c1_init, S, beta, sigma, l_tilde, b_ellip,
           upsilon, chi_n_vec, A, alpha, delta, EulDiff, Eul_Tol,
           c1_options)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.c1_bSp1err()
        hh.get_cnb_vecs()
        aggr.get_K()
        aggr.get_L()
        firms.get_r()
        firms.get_w()

    OBJECTS CREATED WITHIN FUNCTION:
    c1_init    = scalar > 0, initial guess for c1
    S          = integer >= 3, number of periods in individual lifetime
    beta       = scalar in (0,1), discount factor
    sigma      = scalar >= 1, coefficient of relative risk aversion
    l_tilde    = scalar > 0, per-period time endowment for every agent
    b_ellip    = scalar > 0, fitted value of b for elliptical disutility
                 of labor
    upsilon    = scalar > 1, fitted value of upsilon for elliptical
                 disutility of labor
    chi_n_vec  = (S,) vector, values for chi^n_s
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    EulDiff    = boolean, =True if simple difference Euler errors,
                 otherwise percent deviation Euler errors
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    c1_options = length 1 dict, options for c1_bSp1err()
    rpath      = (S,) vector, lifetime path of interest rates
    wpath      = (S,) vector, lifetime path of wages
    c1_args    = length 10 tuple, args to pass into hh.c1_bSp1err()
    results_c1 = results object, results from c1_bSp1err()
    c1         = scalar > 0, optimal initial period consumption given r
                 and w
    cnb_args   = length 8 tuple, args to pass into get_cnb_vecs()
    cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
    nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
    bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
    b_Sp1      = scalar, final period savings, should be close to zero
    n_errors   = (S,) vector, labor supply Euler errors
    b_errors   = (S-1,) vector, savings Euler errors
    K          = scalar > 0, aggregate capital stock
    K_cstr     = boolean, =True if K < epsilon
    L          = scalar > 0, aggregate labor
    L_cstr     = boolean, =True if L < epsilon
    r_params   = length 3 tuple, (A, alpha, delta)
    w_params   = length 2 tuple, (A, alpha)
    r_new      = scalar > 0, guess at steady-state interest rate
    w_new      = scalar > 0, guess at steady-state wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, L, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
             b_errors
    --------------------------------------------------------------------
    '''
    (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A,
        alpha, delta, EulDiff, SS_tol, c1_options) = args

    rpath = r * np.ones(S)
    wpath = w * np.ones(S)
    c1_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
               chi_n_vec, rpath, wpath, EulDiff)
    results_c1 = \
        opt.root(hh.c1_bSp1err, c1_init, args=(c1_args),
                 method='lm', tol=SS_tol, options=(c1_options))
    c1 = results_c1.x
    cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
                EulDiff)
    cvec, nvec, bvec, b_Sp1, n_errors, b_errors = \
        hh.get_cnb_vecs(c1, rpath, wpath, cnb_args)
    K, K_cstr = aggr.get_K(bvec)
    L, L_cstr = aggr.get_L(nvec)
    r_params = (A, alpha, delta)
    r_new = firms.get_r(K, L, r_params)
    w_params = (A, alpha, delta)
    w_new = firms.get_w(r_new, w_params)

    return (K, L, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
            b_errors)


def get_SS(init_vals, args, calibrate_n=False, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply using the bisection method in K
    and L for the outer loop
    --------------------------------------------------------------------
    INPUTS:
    init_vals = length 2 or 3 tuple, (rss_init, c1_init) if calibrate_n=False;
                (rss_init, c1_init, factor_init) if calibrate_n=True
    args      = length 15 or 16 tuple, (S, beta, sigma, l_tilde, b_ellip,
                upsilon, chi_n_vec, A, alpha, delta, Bsct_Tol, Eul_Tol,
                EulDiff, xi, maxiter) if calibrate_n = False;
                (S, beta, sigma, l_tilde, b_ellip, upsilon, A, alpha,
                delta, Bsct_Tol, Factor_tol, Eul_Tol, EulDiff, xi_ss, xi_factor,
                maxiter) if calibrate_n = True
    calibrate_n= boolean, =True if calibrate disutility of labor
    graphs     = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_r()
        firms.get_w()
        utils.print_time()
        inner_loop()
        creat_graphs()
        consumption()
        hrs_by_age()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time = scalar > 0, clock time at beginning of program
    r_init     = scalar > -delta, initial guess for steady-state
                 interest rate
    c1_init    = scalar > 0, initial guess for first period consumpt'n
    factor_init= scalar > 0, initial guess for factor
    S          = integer in [3, 80], number of periods an individual
                 lives
    beta       = scalar in (0,1), discount factor for each model per
    sigma      = scalar > 0, coefficient of relative risk aversion
    l_tilde    = scalar > 0, time endowment for each agent each period
    b_ellip    = scalar > 0, fitted value of b for elliptical disutility
                 of labor
    upsilon    = scalar > 1, fitted value of upsilon for elliptical
                 disutility of labor
    cvec_data  = (S,) vector, consumption by age from 2016 data
    w_data     = scalar, average yearly wage from 2016 data
    y_bar_data = scalar, average household income before tax from 2016 data
    n_data     = (S,) vector, (unit-free) labor supply by age from 2016 data
    chi_n_hat  = (S,) vector, data version of chi_n_vec
    chi_n_vec  = (S,) vector, values for chi^n_s
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    Bsct_Tol   = scalar > 0, tolderance level for outer-loop bisection
                 method
    Eul_Tol    = scalar > 0, tolerance level for inner-loop root finder
    EulDiff    = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                 ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    xi_SS      = scalar in (0, 1], SS updating parameter in outer-loop
                 bisection method
    xi_factor  = scalar in (0, 1], factor updating parameter in outer-loop
    maxiter    = integer >= 1, maximum number of iterations in outer
                 loop bisection method
    iter_SS    = integer >= 0, index of iteration number
    SS_dist    = scalar > 0, distance metric for r in current iteration
    factor_dist= scalar > 0, distance metric for factor in current iteration
    c1_options = length 1 dict, options to pass into
                 opt.root(c1_bSp1err,...)
    rw_params  = length 3 tuple, (A, alpha, delta) args to pass into
                 firms.get_r() and firms.get_w()
    w_init     = scalar, initial value for wage
    inner_args = length 14 tuple, args to pass into inner_loop()
    K_new      = scalar > 0, updated K given r_init, w_init, and bvec
    L_new      = scalar > 0, updated L given r_init, w_init, and nvec
    cvec       = (S,) vector, updated values for lifetime consumption
    nvec       = (S,) vector, updated values for lifetime labor supply
    bvec       = (S,) vector, updated values for lifetime savings
                 (b1, b2,...bS)
    b_Sp1      = scalar, updated value for savings in last period,
                 should be arbitrarily close to zero
    r_new      = scalar > 0, updated interest rate given bvec and nvec
    w_new      = scalar > 0, updated wage given bvec and nvec
    y_dar_model= scalar > 0, average household income from model, given r_init
                 and factor_init
    factor_new = scalar > 0, updated wage given y_bar_data and y_bar_model
    n_errors   = (S,) vector, labor supply Euler errors given r_init
                 and w_init
    b_errors   = (S-1,) vector, savings Euler errors given r_init and
                 w_init
    all_errors = (2S,) vector, (n_errors, b_errors, b_Sp1)
    c_ss       = (S,) vector, steady-state lifetime consumption
    n_ss       = (S,) vector, steady-state lifetime labor supply
    b_ss       = (S,) vector, steady-state wealth enter period with
                 (b1, b2, ...bS)
    b_Sp1_ss   = scalar, steady-state savings for period after last
                 period of life. b_Sp1_ss approx. 0 in equilibrium
    n_err_ss   = (S,) vector, lifetime labor supply Euler errors
    b_err_ss   = (S-1) vector, lifetime savings Euler errors
    r_ss       = scalar > 0, steady-state interest rate
    w_ss       = scalar > 0, steady-state wage
    K_ss       = scalar > 0, steady-state aggregate capital stock
    L_ss       = scalar > 0, steady-state aggregate labor
    Y_params   = length 2 tuple, (A, alpha)
    Y_ss       = scalar > 0, steady-state aggregate output (GDP)
    C_ss       = scalar > 0, steady-state aggregate consumption
    RCerr_ss   = scalar, resource constraint error
    ss_time    = scalar, seconds elapsed for steady-state computation
    ss_output  = length 14 dict, steady-state objects {n_ss, b_ss,
                 c_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
                 n_err_ss, b_err_ss, RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_n.png

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    iter_SS = 0
    SS_dist = 10
    c1_options = {'maxiter': 500}

    if calibrate_n:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        dir = os.path.join(cur_path, 'data')
        (S, beta, sigma, l_tilde, b_ellip, upsilon, A, alpha,
        delta, Bsct_Tol, Factor_tol, Eul_Tol, EulDiff, xi_ss, xi_factor, maxiter) = args
        cvec_data = consump.get_consump(S, 15, 15 + S - 1)
        # average yearly wage and income from data
        w_data = 46662.59
        y_bar_data = 74664
        n_data = hrs.hrs_by_age('dec16', 'dec16', dir, 65, l_tilde = 45)
        n_data = np.concatenate((n_data, np.ones(S - 65) * n_data[-1]))
        r_init, c1_init, factor_init = init_vals
        rw_params = (A, alpha, delta)
        chi_n_hat = (w_data * cvec_data) ** (-sigma)/ \
                        ((b_ellip / l_tilde) * n_data ** (upsilon - 1) * \
                         (1 - (n_data) ** upsilon) ** ((1 - upsilon) / upsilon))
        while (iter_SS < maxiter) and (SS_dist >= Bsct_Tol or factor_dist >= Factor_tol):
            iter_SS += 1
            chi_n_vec = factor_init ** (sigma - 1) * chi_n_hat
            w_init = firms.get_w(r_init, rw_params)
            inner_args = (c1_init, S, beta, sigma, l_tilde, b_ellip,
                          upsilon, chi_n_vec, A, alpha, delta, EulDiff,
                          Eul_Tol, c1_options)
            (K_new, L_new, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
                b_errors) = inner_loop(r_init, w_init, inner_args)
            # Calculate average household income from model
            y_bar_model = (r_init * bvec + w_init * nvec).mean()
            factor_new = y_bar_data / y_bar_model
            all_errors = np.hstack((n_errors, b_errors, b_Sp1))
            SS_dist = (np.absolute(r_new - r_init)).sum()
            factor_dist = np.absolute(factor_new - factor_init)
            r_init = xi_ss * r_new + (1 - xi_ss) * r_init
            factor_init = xi_factor * factor_new + (1 - xi_factor) * factor_init
            print('SS Iter=', iter_SS, ', SS Dist=', '%10.4e' % (SS_dist),
                  ', Factor Dist=', '%10.4e' % (factor_dist), ', Max Abs Err=', '%10.4e' %
                  (np.absolute(all_errors).max()))
    else:
        r_init, c1_init = init_vals
        (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
        delta, Bsct_Tol, Eul_Tol, EulDiff, xi, maxiter) = args
        rw_params = (A, alpha, delta)
        while (iter_SS < maxiter) and (SS_dist >= Bsct_Tol):
            iter_SS += 1
            w_init = firms.get_w(r_init, rw_params)

            inner_args = (c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha, delta, EulDiff, Eul_Tol, c1_options)
            (K_new, L_new, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
                b_errors) = inner_loop(r_init, w_init, inner_args)
            all_errors = np.hstack((n_errors, b_errors, b_Sp1))
            SS_dist = (np.absolute(r_new - r_init)).sum()
            r_init = xi * r_new + (1 - xi) * r_init
            print('SS Iter=', iter_SS, ', SS Dist=', '%10.4e' % (SS_dist),
                  ', Max Abs Err=', '%10.4e' %
                  (np.absolute(all_errors).max()))

    c_ss = cvec.copy()
    n_ss = nvec.copy()
    b_ss = bvec.copy()
    b_Sp1_ss = b_Sp1.copy()
    n_err_ss = n_errors.copy()
    b_err_ss = b_errors.copy()
    r_ss = r_new.copy()
    w_ss = w_new.copy()
    K_ss = K_new.copy()
    L_ss = L_new.copy()
    Y_params = (A, alpha)
    Y_ss = aggr.get_Y(K_ss, L_ss, Y_params)
    C_ss = aggr.get_C(c_ss)
    RCerr_ss = Y_ss - C_ss - delta * K_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'c_ss': c_ss, 'n_ss': n_ss, 'b_ss': b_ss, 'b_Sp1_ss': b_Sp1_ss,
        'w_ss': w_ss, 'r_ss': r_ss, 'K_ss': K_ss, 'L_ss': L_ss,
        'Y_ss': Y_ss, 'C_ss': C_ss, 'n_err_ss': n_err_ss,
        'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
    print('n_ss is: ', n_ss)
    print('b_ss is: ', b_ss)
    print('K_ss=', K_ss, ', L_ss=', L_ss)
    print('r_ss=', r_ss, ', w_ss=', w_ss)
    print('Maximum abs. labor supply Euler error is: ',
          np.absolute(n_err_ss).max())
    print('Maximum abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('Resource constraint error is: ', RCerr_ss)
    print('Steady-state residual savings b_Sp1 is: ', b_Sp1_ss)

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        create_graphs(c_ss, b_ss, n_ss)

    return ss_output


def create_graphs(c_ss, b_ss, n_ss):
    '''
    --------------------------------------------------------------------
    Plot steady-state equilibrium results
    --------------------------------------------------------------------
    INPUTS:
    c_ss = (S,) vector, steady-state lifetime consumption
    b_ss = (S,) vector, steady-state lifetime savings (b1, b2, ...bS)
           where b1 = 0
    n_ss = (S,) vector, steady-state lifetime labor supply

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    S           = integer in [3, 80], number of periods an individual
                  lives
    b_ss_full   = (S+1,) vector, b_ss with zero appended on end.
                  (b1, b2, ...bS, bSp1) where b1, bSp1 = 0
    age_pers_c  = (S,) vector, ages from 1 to S
    age_pers_b  = (S+1,) vector, ages from 1 to S+1

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_n.png

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
    S = n_ss.shape[0]
    b_ss_full = np.append(b_ss, 0)
    age_pers_c = np.arange(1, S + 1)
    age_pers_b = np.arange(1, S + 2)
    fig, ax = plt.subplots()
    plt.plot(age_pers_c, c_ss, marker='D', label='Consumption')
    plt.plot(age_pers_b, b_ss_full, marker='D', label='Savings')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state consumption and savings', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Units of consumption')
    plt.xlim((0, S + 1))
    # plt.ylim((-1.0, 1.15 * (b_ss.max())))
    plt.legend(loc='upper left')
    output_path = os.path.join(output_dir, 'SS_bc')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot steady-state labor supply distributions
    fig, ax = plt.subplots()
    plt.plot(age_pers_c, n_ss, marker='D', label='Labor supply')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Steady-state labor supply', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Labor supply')
    plt.xlim((0, S + 1))
    # plt.ylim((-0.1, 1.15 * (n_ss.max())))
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, 'SS_n')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
