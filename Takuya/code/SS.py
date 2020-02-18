'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and endogenous labor supply from
Chapter 4 of the OG textbook.

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

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def euler_sys(guesses, *args):
    '''
    --------------------------------------------------------------------
    Specify the system of Euler Equations characterizing the household
    problem.
    --------------------------------------------------------------------
    INPUTS:
    guesses = (2S-1,) vector, guess at labor supply and savings decisions
    args = length 14 tuple, (r, w, beta, sigma, l_tilde, chi_n_vec,
            b_ellip, upsilon, diff, S, SS_tol)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.get_n_errors()
        hh.get_n_errors()
        hh.get_cons()

    OBJECTS CREATED WITHIN FUNCTION:
    r    = scalar > 0, guess at steady-state interest rate
    w    = scalar > 0, guess at steady-state wage
    beta       = scalar in (0,1), discount factor
    sigma      = scalar >= 1, coefficient of relative risk aversion
    l_tilde    = scalar > 0, per-period time endowment for every agent
    chi_n_vec  = (S,) vector, values for chi^n_s
    b_ellip    = scalar > 0, fitted value of b for elliptical disutility
                 of labor
    upsilon    = scalar > 1, fitted value of upsilon for elliptical
                 disutility of labor
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    diff    = boolean, =True if simple difference Euler errors,
                 otherwise percent deviation Euler errors
    S          = integer >= 3, number of periods in individual lifetime
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    nvec       = (S,) vector, lifetime labor supply (n1, n2, ...nS)
    bvec       = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
    cvec       = (S,) vector, lifetime consumption (c1, c2, ...cS)
    b_sp1      = = (S,) vector, lifetime savings (b1, b2, ...bS) with bS=0
    n_errors   = (S,) vector, labor supply Euler errors
    b_errors   = (S-1,) vector, savings Euler errors

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: array of n_errors and b_errors
    --------------------------------------------------------------------
    '''
    (r, w, beta, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, diff, S,
     SS_tol) = args
    nvec = guesses[:S]
    bvec1 = guesses[S:]

    bvec = np.append(0.0, bvec1)
    b_sp1 = np.append(bvec[1:], 0.0)
    cvec = hh.get_cons(r, w, bvec, b_sp1, nvec)
    n_args = (w, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, diff,
              cvec)
    n_errors = hh.get_n_errors(nvec, *n_args)
    b_args = (r, beta, sigma, diff)
    b_errors = hh.get_b_errors(cvec, *b_args)

    errors = np.append(n_errors, b_errors)

    return errors


def inner_loop(r, w, args):
    '''
    --------------------------------------------------------------------
    Given values for r and w, solve for the households' optimal decisions
    --------------------------------------------------------------------
    INPUTS:
    r    = scalar > 0, guess at steady-state interest rate
    w    = scalar > 0, guess at steady-state wage
    args = length 14 tuple, (nvec_init, bvec_init, S, beta, sigma,
            l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha, delta,
            EulDiff, SS_tol)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        euler_sys()
        aggr.get_K()
        aggr.get_L()
        firms.get_r()
        firms.get_w()

    OBJECTS CREATED WITHIN FUNCTION:
    nvec_init  = (S,) vector, initial guesses at choice of labor supply
    bvec_init  = (S,) vector, initial guesses at choice of savings
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
    (nvec_init, bvec_init, S, beta, sigma, l_tilde, b_ellip, upsilon,
     chi_n_vec, A, alpha, delta, EulDiff, SS_tol) = args

    euler_args = (r, w, beta, sigma, l_tilde, chi_n_vec, b_ellip,
                  upsilon, EulDiff, S, SS_tol)
    guesses = np.append(nvec_init, bvec_init[1:])
    results_euler = opt.root(euler_sys, guesses, args=(euler_args),
                             method='lm', tol=SS_tol)
    nvec = results_euler.x[:S]
    bvec = np.append(0.0, results_euler.x[S:])
    n_errors = results_euler.fun[:S]
    b_errors = results_euler.fun[S:]
    b_sp1 = np.append(bvec[1:], 0.0)
    cvec = hh.get_cons(r, w, bvec, b_sp1, nvec)
    b_Sp1 = 0.0
    K, K_cstr = aggr.get_K(bvec)
    L, L_cstr = aggr.get_L(nvec)
    r_params = (A, alpha, delta)
    r_new = firms.get_r(K, L, r_params)
    w_params = (A, alpha, delta)
    w_new = firms.get_w(r_new, w_params)

    return (K, L, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
            b_errors)


def get_SS(init_vals, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply using the bisection method in K
    and L for the outer loop
    --------------------------------------------------------------------
    INPUTS:
    init_vals = length 2 tuple, (rss_init, c1_init)
    args      = length 15 tuple, (S, beta, sigma, l_tilde, b_ellip,
                upsilon, chi_n_vec, A, alpha, delta, Bsct_Tol, Eul_Tol,
                EulDiff, xi, maxiter)
    graphs    = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_r()
        firms.get_w()
        utils.print_time()
        inner_loop()
        creat_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time = scalar > 0, clock time at beginning of program
    r_init     = scalar > -delta, initial guess for steady-state
                 interest rate
    c1_init    = scalar > 0, initial guess for first period consumpt'n
    S          = integer in [3, 80], number of periods an individual
                 lives
    beta       = scalar in (0,1), discount factor for each model per
    sigma      = scalar > 0, coefficient of relative risk aversion
    l_tilde    = scalar > 0, time endowment for each agent each period
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
    Bsct_Tol   = scalar > 0, tolderance level for outer-loop bisection
                 method
    Eul_Tol    = scalar > 0, tolerance level for inner-loop root finder
    EulDiff    = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                 ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    xi         = scalar in (0, 1], SS updating parameter in outer-loop
                 bisection method
    maxiter    = integer >= 1, maximum number of iterations in outer
                 loop bisection method
    iter_SS    = integer >= 0, index of iteration number
    dist       = scalar > 0, distance metric for current iteration
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
    r_init, c1_init = init_vals
    (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
     delta, Bsct_Tol, Eul_Tol, EulDiff, xi, maxiter) = args
    iter_SS = 0
    dist = 10
    nvec_init = np.ones(S) * 0.4
    bvec_init = np.append(0.0, np.ones(S - 1) * 0.05)
    rw_params = (A, alpha, delta)
    while (iter_SS < maxiter) and (dist >= Bsct_Tol):
        iter_SS += 1
        w_init = firms.get_w(r_init, rw_params)
        inner_args = (nvec_init, bvec_init, S, beta, sigma, l_tilde,
                      b_ellip, upsilon, chi_n_vec, A, alpha, delta,
                      EulDiff, Eul_Tol)
        (K_new, L_new, cvec, nvec, bvec, b_Sp1, r_new, w_new, n_errors,
            b_errors) = inner_loop(r_init, w_init, inner_args)
        all_errors = np.hstack((n_errors, b_errors, b_Sp1))
        dist = (np.absolute(r_new - r_init)).sum()
        r_init = xi * r_new + (1 - xi) * r_init
        print('SS Iter=', iter_SS, ', SS Dist=', '%10.4e' % (dist),
              ', Max Abs Err=', '%10.4e' %
              (np.absolute(all_errors).max()))

    c_ss = cvec
    n_ss = nvec
    b_ss = bvec
    b_Sp1_ss = b_Sp1
    n_err_ss = n_errors
    b_err_ss = b_errors
    r_ss = r_new
    w_ss = w_new
    K_ss = K_new
    L_ss = L_new
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
