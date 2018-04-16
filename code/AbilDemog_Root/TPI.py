'''
------------------------------------------------------------------------
This module contains the functions used to solve the transition path
equilibrium using time path iteration (TPI) for the OG-ITA overlapping
generations model with S-period lived agents, endogenous labor supply,
heterogeneous ability, demographics, growth, and bequests.

This Python module imports the following module(s):
    aggregates.py
    firms.py
    households.py
    utilities.py

This Python module defines the following function(s):
    get_path()
    get_cnbpath()
    get_TPI()
    create_graphs()
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import aggregates as aggr
import firms
import households as hh
import utilities as utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_path(x1, xT, T, spec):
    '''
    --------------------------------------------------------------------
    This function generates a path from point x1 to point xT such that
    that the path x is a linear or quadratic function of time t.

        linear:    x = d*t + e
        quadratic: x = a*t^2 + b*t + c

    The identifying assumptions for quadratic are the following:

        (1) x1 is the value at time t=0: x1 = c
        (2) xT is the value at time t=T-1: xT = a*(T-1)^2 + b*(T-1) + c
        (3) the slope of the path at t=T-1 is 0: 0 = 2*a*(T-1) + b
    --------------------------------------------------------------------
    INPUTS:
    x1 = scalar, initial value of the function x(t) at t=0
    xT = scalar, value of the function x(t) at t=T-1
    T  = integer >= 3, number of periods of the path
    spec = string, "linear" or "quadratic"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    cc    = scalar, constant coefficient in quadratic function
    bb    = scalar, coefficient on t in quadratic function
    aa    = scalar, coefficient on t^2 in quadratic function
    xpath = (T,) vector, parabolic xpath from x1 to xT

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: xpath
    --------------------------------------------------------------------
    '''
    if spec == 'linear':
        xpath = np.linspace(x1, xT, T)
    elif spec == 'quadratic':
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = (aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) +
                 cc)

    return xpath


def get_cnbpath(rpath, wpath, BQpath, args):
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
    (J, S, T2, lambdas, emat, zeta_mat, omega_path, mort_rates,
        chi_n_vec, chi_b_vec, beta, sigma, l_tilde, b_ellip, upsilon,
        g_y, bmat1, n_ss, In_Tol) = args
    cpath = np.zeros((S, J, T2 + S - 1))
    npath = np.zeros((S, J, T2 + S - 1))
    bpath = np.zeros((S, J, T2 + S))
    bpath[:, :, 0] = bmat1
    n_err_path = np.zeros((S, J, T2 + S - 1))
    b_err_path = np.zeros((S, J, T2 + S))

    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    for per_rmn in range(1, S):

        for j_ind in range(J):

            if per_rmn == 1:
                # per_rmn=1 individual only has an s=S labor supply
                # decision n_S
                bn_S1_init = np.array([bmat1[-1, j_ind],
                                       n_ss[-1, j_ind]])
                bnS1_args = \
                    (rpath[0], wpath[0], BQpath[0], bmat1[-2, j_ind],
                     lambdas[j_ind], emat[-1, j_ind],
                     zeta_mat[-1, j_ind], omega_path[0, -1],
                     mort_rates[-1], per_rmn, beta, sigma, l_tilde,
                     chi_n_vec[-1], chi_b_vec[j_ind], b_ellip, upsilon,
                     g_y)
                results_bnS1 = opt.root(hh.bn_errors, bn_S1_init,
                                        args=(bnS1_args), method='lm',
                                        tol=In_Tol)
                b_S1, n_S1 = results_bnS1.x
                npath[-1, j_ind, 0] = n_S1
                bpath[-1, j_ind, 1] = b_S1
                b_err_path[-1, j_ind, 1], n_err_path[-1, j_ind, 0] = \
                    results_bnS1.fun
                c_args = (emat[-1, j_ind], zeta_mat[-1, j_ind],
                          lambdas[j_ind], omega_path[0, -1], g_y)
                cpath[-1, j_ind, 0] = \
                    hh.get_cons(rpath[0], wpath[0], bmat1[-2, j_ind],
                                b_S1, n_S1, BQpath[0], c_args)
                # print('per_rmn=', per_rmn, 'j=', j_ind,
                #       ', Max Err=', "%10.4e" %
                #       np.absolute(results_bnS1.fun).max())
                # if np.absolute(results_bnS1.fun).max() > 1e-10:
                #     print(results_bnS1.success)
            else:
                # 1<p<S chooses b_{s+1} and n_s and has incomplete lives
                DiagMaskbn = np.eye(per_rmn, dtype=bool)
                b_sp1_init = np.diag(bpath[-per_rmn:, j_ind, :per_rmn])
                n_s_init = \
                    np.hstack((n_ss[S - per_rmn, j_ind],
                               np.diag(npath[S - per_rmn + 1:, j_ind,
                                             :per_rmn - 1])))
                bn_init = np.hstack((b_sp1_init, n_s_init))
                bn_args = \
                    (rpath[:per_rmn], wpath[:per_rmn], BQpath[:per_rmn],
                     bmat1[-(per_rmn + 1), j_ind], lambdas[j_ind],
                     emat[-per_rmn:, j_ind], zeta_mat[-per_rmn, j_ind],
                     np.diag(omega_path[:per_rmn, -per_rmn:]),
                     mort_rates[-per_rmn:], per_rmn, beta, sigma,
                     l_tilde, chi_n_vec[-per_rmn:], chi_b_vec[j_ind],
                     b_ellip, upsilon, g_y)
                results_bn = opt.root(hh.bn_errors, bn_init,
                                      args=(bn_args), method='lm',
                                      tol=In_Tol)
                bvec = results_bn.x[:per_rmn]
                nvec = results_bn.x[per_rmn:]
                b_errors = results_bn.fun[:per_rmn]
                n_errors = results_bn.fun[per_rmn:]
                b_s_vec = np.append(bmat1[-(per_rmn + 1), j_ind],
                                    bvec[:-1])
                b_sp1_vec = bvec.copy()
                c_args = (emat[-per_rmn:, j_ind],
                          zeta_mat[-per_rmn:, j_ind], lambdas[j_ind],
                          np.diag(omega_path[:per_rmn, -per_rmn:]), g_y)
                cvec = hh.get_cons(rpath[:per_rmn], wpath[:per_rmn],
                                   b_s_vec, b_sp1_vec, nvec,
                                   BQpath[:per_rmn], c_args)
                npath[-per_rmn:, j_ind, :per_rmn] = \
                    DiagMaskbn * nvec + npath[-per_rmn:, j_ind,
                                              :per_rmn]
                bpath[-per_rmn:, j_ind, 1:per_rmn + 1] = \
                    (DiagMaskbn * bvec + bpath[-per_rmn:, j_ind,
                                               1:per_rmn + 1])
                cpath[-per_rmn:, j_ind, :per_rmn] = \
                    DiagMaskbn * cvec + cpath[-per_rmn:, j_ind,
                                              :per_rmn]
                n_err_path[-per_rmn:, j_ind, :per_rmn] = \
                    (DiagMaskbn * n_errors +
                     n_err_path[-per_rmn:, j_ind, :per_rmn])
                b_err_path[-per_rmn:, j_ind, 1:per_rmn + 1] = \
                    (DiagMaskbn * b_errors +
                     b_err_path[-per_rmn:, j_ind, 1:per_rmn + 1])
                # print('per_rmn=', per_rmn, 'j=', j_ind,
                #       ', Max Err=', "%10.4e" %
                #       results_bn.fun.max())
                # if np.absolute(results_bn.fun).max() > 1e-10:
                #     print(results_bn.success)

    # Solve the complete remaining lifetime decisions of agents born
    # between period t=1 and t=T2
    DiagMaskbn = np.eye(S, dtype=bool)
    # print('bpath, j_ind==0', bpath[:5, 0, :5])
    # print('bpath, j_ind==2', bpath[:5, 2, :5])
    # print('bpath, j_ind==4', bpath[:5, 4, :5])
    # print('npath, j_ind==0', npath[:5, 0, :5])
    # print('npath, j_ind==2', npath[:5, 2, :5])
    # print('npath, j_ind==4', npath[:5, 4, :5])
    # print('n_ss[0, 2], j_ind=2', n_ss[0, 2])

    for t in range(T2):

        for j_ind in range(J):

            if t == 0:
                # In period t=0, make initial guess vectors a little
                # more feasible by decreasing bvec by 10% and increasing
                # nvec by 15%
                b_sp1_init = 0.9 * np.diag(bpath[:, j_ind, t:t + S])
                n1_init = np.diag(npath[1:, j_ind, t:t + S - 1])[0]
                n2_init = np.diag(npath[1:, j_ind, t:t + S - 1])[1]
                n_s_init = 1.1 * np.append(2 * n1_init - n2_init,
                                           np.diag(npath[1:, j_ind,
                                                         t:t + S - 1]))
                # if j_ind == 2:
                #     print('bpath, j_ind=1', np.diag(bpath[:, 1, :S]))
                #     print('npath, j_ind=1', np.diag(bpath[:, 1, 1:S + 1]))
                #     print('t=', t, ', j=', j_ind, 'n_s_init=', n_s_init)
                #     print('t=', t, ', j=', j_ind, 'b_sp1_init=', b_sp1_init)
                # n_s_init = np.append(n_ss[0, j_ind],
                #                      np.diag(npath[1:, j_ind,
                #                                    t:t + S - 1]))
            else:
                b_sp1_init = np.diag(bpath[:, j_ind, t:t + S])
                n_s_init = np.diag(npath[:, j_ind, t - 1:t + S - 1])
            bn_init = np.hstack((b_sp1_init, n_s_init))
            bn_args = (rpath[t:t + S], wpath[t:t + S], BQpath[t:t + S],
                       0.0, lambdas[j_ind], emat[:, j_ind],
                       zeta_mat[:, j_ind],
                       np.diag(omega_path[t:t + S, :]), mort_rates, S,
                       beta, sigma, l_tilde, chi_n_vec,
                       chi_b_vec[j_ind], b_ellip, upsilon, g_y)

            results_bn = opt.root(hh.bn_errors, bn_init, args=(bn_args),
                                  method='lm', tol=In_Tol)
            bvec = results_bn.x[:S]
            nvec = results_bn.x[S:]
            b_errors = results_bn.fun[:S]
            n_errors = results_bn.fun[S:]
            b_s_vec = np.append(0.0, bvec[:-1])
            b_sp1_vec = bvec.copy()
            c_args = (emat[:, j_ind], zeta_mat[:, j_ind],
                      lambdas[j_ind], np.diag(omega_path[t:t + S, :]),
                      g_y)
            cvec = hh.get_cons(rpath[t:t + S], wpath[t:t + S], b_s_vec,
                               b_sp1_vec, nvec, BQpath[t:t + S], c_args)
            # if t == 0 and j_ind == 2:
            #     print('bvec=', bvec)
            #     print('nvec=', nvec)
            #     print('cvec=', cvec)
            #     print('b_errors=', b_errors)
            #     print('n_errors=', n_errors)
            npath[:, j_ind, t:t + S] = (DiagMaskbn * nvec +
                                        npath[:, j_ind, t:t + S])
            bpath[:, j_ind, t + 1:t + S + 1] = \
                DiagMaskbn * bvec + bpath[:, j_ind, t + 1:t + S + 1]
            cpath[:, j_ind, t:t + S] = (DiagMaskbn * cvec +
                                        cpath[:, j_ind, t:t + S])
            n_err_path[:, j_ind, t:t + S] = \
                DiagMaskbn * n_errors + n_err_path[:, j_ind, t:t + S]
            b_err_path[:, j_ind, t + 1:t + S + 1] = \
                (DiagMaskbn * b_errors +
                 b_err_path[:, j_ind, t + 1:t + S + 1])
            # print('t=', t, 'j=', j_ind, ', Max Err=', "%10.4e" %
            #       results_bn.fun.max())
            # if np.absolute(results_bn.fun).max() > 1e-10:
            #     print(results_bn.success)
            #     print('b=', bvec, ', n=', nvec, ', b_err=', b_errors,
            #           ', n_err=', n_errors, ', c=', cvec)

    return cpath, npath, bpath, n_err_path, b_err_path


def get_TPI(bmat1, args, graphs):
    '''
    --------------------------------------------------------------------
    Solves for transition path equilibrium using time path iteration
    (TPI)
    --------------------------------------------------------------------
    INPUTS:
    bmat1 = (S, J) matrix, initial period wealth (savings) distribution
    args  = length 25 tuple, (J, S, T1, T2, lambdas, emat, beta, sigma,
            l_tilde, b_ellip, upsilon, chi_n_vec, Z, gamma, delta, r_ss,
            K_ss, L_ss, C_ss, b_ss, n_ss, maxiter, Out_Tol, In_Tol, xi)
    graphs = Boolean, =True if want graphs of TPI objects

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_path()
        firms.get_r()
        firms.get_w()
        get_cnbpath()
        aggr.get_K()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        utils.print_time()
        create_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time   = scalar, current processor time in seconds (float)
    J            = integer >= 1, number of heterogeneous ability groups
    S            = integer in [3,80], number of periods an individual
                   lives
    T1           = integer > S, number of time periods until steady
                   state is assumed to be reached
    T2           = integer > T1, number of time periods after which
                   steady-state is forced in TPI
    lambdas      = (J,) vector, income percentiles for distribution of
                   ability within each cohort
    emat         = (S, J) matrix, lifetime ability profiles for each
                   lifetime income group j
    beta         = scalar in (0,1), discount factor for model period
    sigma        = scalar > 0, coefficient of relative risk aversion
    l_tilde      = scalar > 0, time endowment for each agent each
                   period
    b_ellip      = scalar > 0, fitted value of b for elliptical
                   disutility of labor
    upsilon      = scalar > 1, fitted value of upsilon for elliptical
                   disutility of labor
    chi_n_vec    = (S,) vector, values for chi^n_s
    Z            = scalar > 0, total factor productivity parameter in
                   firms' production function
    gamma        = scalar in (0,1), capital share of income
    delta        = scalar in [0,1], per-period capital depreciation rt
    r_ss         = scalar > 0, steady-state aggregate interest rate
    K_ss         = scalar > 0, steady-state aggregate capital stock
    L_ss         = scalar > 0, steady-state aggregate labor
    C_ss         = scalar > 0, steady-state aggregate consumption
    b_ss         = (S,) vector, steady-state savings distribution
                   (b1, b2,... bS)
    n_ss         = (S,) vector, steady-state labor supply distribution
                   (n1, n2,... nS)
    maxiter      = integer >= 1, Maximum number of iterations for TPI
    Out_Tol      = scalar > 0, convergence criterion for TPI outer loop
    In_Tol       = scalar > 0, tolerance level for TPI inner-loop root
                   finders
    xi           = scalar in (0,1], TPI path updating parameter
    K1           = scalar > 0, initial aggregate capital stock
    K1_cnstr     = Boolean, =True if K1 <= 0
    rpath_init   = (T2+S-1,) vector, initial guess for the time path of
                   interest rates
    iter_TPI     = integer >= 0, current iteration of TPI
    dist         = scalar >= 0, distance measure between initial and
                   new paths
    rw_params    = length 3 tuple, (Z, gamma, delta)
    Y_params     = length 2 tuple, (Z, gamma)
    cnb_args     = length 13 tuple, args to pass into get_cnbpath()
    rpath        = (T2+S-1,) vector, time path of the interest rates
    wpath        = (T2+S-1,) vector, time path of the wages
    cpath        = (S, J, T2+S-1) array, time path of distribution of
                   household consumption c_{j,s,t}
    npath        = (S, J, T2+S-1) matrix, time path of distribution of
                   household labor supply n_{j,s,t}
    bpath        = (S, J, T2+S-1) matrix, time path of distribution of
                   household savings b_{j,s,t}
    n_err_path   = (S, J, T2+S-1) matrix, time path of distribution of
                   household labor supply Euler errors
    b_err_path   = (S, J, T2+S-1) matrix, time path of distribution of
                   household savings Euler errors. b_{j,1,t} = 0 for all
                   j and t
    Kpath_new    = (T2+S-1,) vector, new path of aggregate capital stock
                   implied by household and firm optimization
    Kpath_cnstr  = (T2+S-1,) Boolean vector, =True if K_t<=epsilon
    Lpath_new    = (T2+S-1,) vector, new path of aggregate labor implied
                   by household and firm optimization
    Lpath_cstr   = (T2+S-1,) Boolean vector, =True if L_t<=epsilon
    rpath_new    = (T2+S-1,) vector, updated time path of interest rate
    Ypath        = (T2+S-1,) vector, equilibrium time path of aggregate
                   output (GDP) Y_t
    Cpath        = (T2+S-1,) vector, equilibrium time path of aggregate
                   consumption C_t
    RCerrPath    = (T2+S-2,) vector, equilibrium time path of the
                   resource constraint error. Should be 0 in eqlb'm
                   Y_t - C_t - K_{t+1} + (1-delta)*K_t
    MaxAbsEulErr = scalar > 0, maximum absolute Euler error. Includes
                   both labor supply and savings Euler errors
    Kpath        = (T2+S-1,) vector, equilibrium time path of aggregate
                   capital stock K_t
    Lpath        = (T2+S-1,) vector, equilibrium time path of aggregate
                   labor L_t
    tpi_time     = scalar, time to compute TPI solution (seconds)
    tpi_output   = length 13 dictionary, {cpath, npath, bpath, wpath,
                   rpath, Kpath, Lpath, Ypath, Cpath, n_err_path,
                   b_err_path, RCerrPath, tpi_time}
    graph_args   = length 4 tuple, (J, S, T2, lambdas)

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        rpath.png
        wpath.png
        Ypath.png
        C_aggr_path.png
        cpath_avg_s.png
        cpath_avg_j.png
        npath_avg_s.png
        npath_avg_j.png
        bpath_avg_s.png
        bpath_avg_j.png

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    (J, E, S, T1, T2, lambdas, emat, mort_rates, imm_rates_mat,
        omega_path, omega_preTP, g_n_path, zeta_mat, chi_n_vec,
        chi_b_vec, beta, sigma, l_tilde, b_ellip, upsilon, g_y, Z,
        gamma, delta, r_ss, BQ_ss, K_ss, K1, L_ss, NX_ss, C_ss, b_ss,
        n_ss, maxiter, Out_Tol, In_Tol, xi) = args

    # Create time paths for r and BQ
    rpath_init = np.zeros(T2 + S - 1)
    BQpath_init = np.zeros(T2 + S - 1)
    rpath_init[:T1] = get_path(r_ss, r_ss, T1, 'quadratic')
    rpath_init[T1:] = r_ss
    BQ_args = (lambdas, omega_path[0, :], mort_rates, g_n_path[0])
    BQ_1 = aggr.get_BQ(bmat1, r_ss, BQ_args)
    BQpath_init[:T1] = get_path(BQ_1, BQ_ss, T1, 'quadratic')
    BQpath_init[T1:] = BQ_ss

    iter_TPI = int(0)
    dist = 10.0
    rw_params = (Z, gamma, delta)
    Y_params = (Z, gamma)
    cnb_args = (J, S, T2, lambdas, emat, zeta_mat, omega_path,
                mort_rates, chi_n_vec, chi_b_vec, beta, sigma, l_tilde,
                b_ellip, upsilon, g_y, bmat1, n_ss, In_Tol)

    while (iter_TPI < maxiter) and (dist >= Out_Tol):
        iter_TPI += 1
        rpath = rpath_init
        BQpath = BQpath_init
        wpath = firms.get_w(rpath_init, rw_params)
        cpath, npath, bpath, n_err_path, b_err_path = \
            get_cnbpath(rpath, wpath, BQpath, cnb_args)

        Kpath_new = np.zeros(T2 + S - 1)
        Kpath_new[0] = K1
        K_args = (lambdas, omega_path[:T2 - 1, :],
                  imm_rates_mat[:T2 - 1, :], g_n_path[1:T2])
        Kpath_new[1:T2], Kpath_cstr = aggr.get_K(bpath[:, :, 1:T2],
                                                 K_args)
        Kpath_new[T2:] = K_ss
        Kpath_cstr = np.append(Kpath_cstr, np.zeros(S - 1, dtype=bool))

        Lpath_new = np.zeros(T2 + S - 1)
        L_args = (emat, lambdas, omega_path[:T2, :])
        Lpath_new[:T2], Lpath_cstr = aggr.get_L(npath[:, :, :T2],
                                                L_args)
        Lpath_new[T2:] = L_ss
        Lpath_cstr = np.append(Lpath_cstr, np.zeros(S - 1, dtype=bool))

        rpath_new = firms.get_r(Kpath_new, Lpath_new, rw_params)

        BQpath_new = np.zeros(T2 + S - 1)
        omega_path0 = np.vstack((omega_preTP.reshape((1, S)),
                                 omega_path[:T2 - 1, :]))
        BQ_args = (lambdas, omega_path0, mort_rates, g_n_path[:T2])
        BQpath_new[:T2] = aggr.get_BQ(bpath[:, :, :T2], rpath_new[:T2],
                                      BQ_args)
        BQpath_new[T2:] = BQ_ss

        Ypath = aggr.get_Y(Kpath_new, Lpath_new, Y_params)
        Cpath = np.zeros(T2 + S - 1)
        C_args = (lambdas, omega_path[:T2, :])
        Cpath[:T2] = aggr.get_C(cpath[:, :, :T2], C_args)
        Cpath[T2:] = C_ss

        NXpath = np.zeros(T2 + S - 1)
        NX_args = (lambdas, omega_path[1:T2 + 1, :],
                   imm_rates_mat[1:T2 + 1, :], g_y)
        NXpath[:T2] = aggr.get_NX(bpath[:, :, 1:T2 + 1], NX_args)
        NXpath[T2:] = NX_ss

        RCerrPath = np.zeros(T2 + S - 1)
        RCerrPath[:-1] = \
            (Ypath[:-1] - Cpath[:-1] -
             (np.exp(g_y) * (1 + g_n_path[1:]) * Kpath_new[1:]) +
             (1 - delta) * Kpath_new[:-1] - NXpath[:-1])
        RCerrPath[-1] = RCerrPath[-2]

        # Check the distance of rpath_new and BQpath_new
        rBQpath_init = np.append(rpath_init[:T2], BQpath_init[:T2])
        rBQpath_new = np.append(rpath_new[:T2], BQpath_new[:T2])
        dist = (np.absolute(rBQpath_new - rBQpath_init)).max()
        MaxAbsEulErr = \
            np.absolute(np.append(n_err_path.flatten(),
                                  b_err_path.flatten())).max()
        print('TPI iter: ', iter_TPI, ', dist: ', "%10.4e" % (dist),
              ', max abs all errs: ', "%10.4e" % MaxAbsEulErr)
        # The resource constraint does not bind across the transition
        # path until the equilibrium is solved
        rpath_init[:T2] = (xi * rpath_new[:T2] +
                           (1 - xi) * rpath_init[:T2])
        rpath_init[T2:] = r_ss
        BQpath_init[:T2] = (xi * BQpath_new[:T2] +
                            (1 - xi) * BQpath_init[:T2])
        BQpath_init[T2:] = BQ_ss

    if (iter_TPI == maxiter) and (dist > Out_Tol):
        print('TPI reached maxiter and did not converge.')
    elif (iter_TPI == maxiter) and (dist <= Out_Tol):
        print('TPI converged in the last iteration. ' +
              'Should probably increase maxiter_TPI.')
    Kpath = Kpath_new
    Lpath = Lpath_new
    BQpath = BQpath_init

    tpi_time = time.clock() - start_time

    tpi_output = {
        'cpath': cpath, 'npath': npath, 'bpath': bpath, 'wpath': wpath,
        'rpath': rpath, 'Kpath': Kpath, 'Lpath': Lpath, 'Ypath': Ypath,
        'Cpath': Cpath, 'BQpath': BQpath, 'NXpath': NXpath,
        'n_err_path': n_err_path, 'b_err_path': b_err_path,
        'RCerrPath': RCerrPath, 'tpi_time': tpi_time}

    # Print maximum resource constraint error. Only look at resource
    # constraint up to period T2 - 1 because period T2 includes K_{t+1},
    # which was forced to be the steady-state
    print('Max abs. RC error: ', "%10.4e" %
          (np.absolute(RCerrPath[:T2 - 1]).max()))

    # Print TPI computation time
    utils.print_time(tpi_time, 'TPI')

    if graphs:
        graph_args = (J, S, T2, lambdas)
        create_graphs(tpi_output, graph_args)

    return tpi_output


def create_graphs(tpi_output, args):
    '''
    --------------------------------------------------------------------
    Plot equilibrium time path results
    --------------------------------------------------------------------
    INPUTS:
    tpi_output = length 13 dict, equilibrium time paths and computation
                 time from TPI computation
    args       = length 4 tuple, (J, S, T2, lambdas)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    J           = integer >= 1, number of heterogeneous ability groups
    S           = integer in [3,80], number of periods an individual
                  lives
    T2          = integer > 0, number of time periods after which steady
                  state is forced in TPI
    lambdas     = (J,) vector, income percentiles for distribution of
                  ability within each cohort
    Kpath       = (T2+S-1,) vector, time path of aggregate capital stock
    Lpath       = (T2+S-1,) vector, time path of aggregate labor
    rpath       = (T2+S-1,) vector, time path of interest rate
    wpath       = (T2+S-1,) vector, time path of wage
    Ypath       = (T2+S-1,) vector, time path of aggregate output (GDP)
    Cpath       = (T2+S-1,) vector, time path of aggregate consumption
    cpath       = (S, J, T2+S-1) array, time path of distribution of
                  household consumption
    npath       = (S, J, T2+S-1) array, time path of distribution of
                  household labor supply
    bpath       = (S, J, T2+S-1) array, time path of distribution of
                  household savings
    cur_path    = string, path name of current directory
    images_fldr = string, folder in current path to save files
    images_dir  = string, total path of images folder
    images_path = string, path of file name of figure to be saved
    tvec        = (T2+S-1,) vector, time period vector
    lambdas_arr = (S, J, T2) array, lambdas row vector copied down S
                  rows and back T2 rows into the 3rd dimension
    tgridT      = (T2,) vector, time period vector from 1 to T2
    sgrid       = (S,) vector, all ages from 1 to S
    lamcumsum   = (J,) vector, cumulative sum of lambdas percentages
    jmidgrid    = (J,) vector, midpoints of lamcumsum percentile bins
    tmat_s      = (S, T2) matrix, time periods for decisions ages
                  (S) and time periods (T2)
    smat        = (S, T2) matrix, ages for all decisions ages (S)
                  and time periods (T2)
    tmat_j      = (J, T2) matrix, time periods for decisions abilities
                  (J) and time periods (T2)
    jmat        = (J, T2) matrix, abilities for all decisions abilities
                  (J) and time periods (T2)
    cmap_c      = color map for household consumption plots
    cmap_n      = color map for household labor supply plots
    cmap_b      = color map for household savings plots
    cpath_avg_s = (S, T2) matrix, time path of distribution of household
                  consumption averaged over ability types
    cpath_avg_j = (J, T2) matrix, time path of distribution of household
                  consumption averaged over ages
    npath_avg_s = (S, T2) matrix, time path of distribution of household
                  labor supply averaged over ability types
    npath_avg_j = (J, T2) matrix, time path of distribution of household
                  labor supply averaged over ages
    bpath_avg_s = (S, T2) matrix, time path of distribution of household
                  savings averaged over ability types
    bpath_avg_j = (J, T2) matrix, time path of distribution of household
                  savings averaged over ages

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        rpath.png
        wpath.png
        Ypath.png
        C_aggr_path.png
        cpath_avg_s.png
        cpath_avg_j.png
        npath_avg_s.png
        npath_avg_j.png
        bpath_avg_s.png
        bpath_avg_j.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    J, S, T2, lambdas = args
    Kpath = tpi_output['Kpath']
    Lpath = tpi_output['Lpath']
    rpath = tpi_output['rpath']
    wpath = tpi_output['wpath']
    Ypath = tpi_output['Ypath']
    Cpath = tpi_output['Cpath']
    cpath = tpi_output['cpath']
    npath = tpi_output['npath']
    bpath = tpi_output['bpath']

    # Create TPI images directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    images_fldr = 'images/TPI'
    images_dir = os.path.join(cur_path, images_fldr)
    if not os.access(images_dir, os.F_OK):
        os.makedirs(images_dir)

    # Plot time path of aggregate capital stock
    tvec = np.linspace(1, T2 + S - 1, T2 + S - 1)
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    plt.plot(tvec, Kpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate capital stock K')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate capital $K_{t}$')
    images_path = os.path.join(images_dir, 'Kpath')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate capital stock
    fig, ax = plt.subplots()
    plt.plot(tvec, Lpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate labor L')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate labor $L_{t}$')
    images_path = os.path.join(images_dir, 'Lpath')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate output (GDP)
    fig, ax = plt.subplots()
    plt.plot(tvec, Ypath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate output (GDP) Y')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate output $Y_{t}$')
    images_path = os.path.join(images_dir, 'Ypath')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of aggregate consumption
    fig, ax = plt.subplots()
    plt.plot(tvec, Cpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for aggregate consumption C')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate consumption $C_{t}$')
    images_path = os.path.join(images_dir, 'C_aggr_path')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of wage
    fig, ax = plt.subplots()
    plt.plot(tvec, wpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for real wage w')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Wage $w_{t}$')
    images_path = os.path.join(images_dir, 'wpath')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of interest rate
    fig, ax = plt.subplots()
    plt.plot(tvec, rpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for real interest rate r')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Interest rate $r_{t}$')
    images_path = os.path.join(images_dir, 'rpath')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of individual consumption distribution averaged
    # over ability j
    lambdas_arr = np.tile(lambdas.reshape((1, J, 1)), (S, 1, T2))
    cpath_avg_s = (cpath[:, :, :T2] * lambdas_arr).sum(axis=1)
    tgridT = np.linspace(1, T2, T2)
    sgrid = np.linspace(1, S, S)
    tmat_s, smat = np.meshgrid(tgridT, sgrid)
    cmap_c = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'average consumption $c_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat_s, smat, cpath_avg_s, rstride=strideval,
                    cstride=strideval, cmap=cmap_c)
    images_path = os.path.join(images_dir, 'cpath_avg_s')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of individual consumption distribution averaged
    # over age s
    pop_dist_age = (1 / S) * np.ones((S, J, T2))
    cpath_avg_j = (cpath[:, :, :T2] * pop_dist_age).sum(axis=0)
    lamcumsum = lambdas.cumsum()
    jmidgrid = 0.5 * lamcumsum + 0.5 * (lamcumsum - lambdas)
    tmat_j, jmat = np.meshgrid(tgridT, jmidgrid)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'ability-$j$')
    ax.set_zlabel(r'average consumption $c_{j,t}$')
    strideval = max(int(1), int(round(J / 10)))
    ax.plot_surface(tmat_j, jmat, cpath_avg_j, rstride=strideval,
                    cstride=strideval, cmap=cmap_c)
    images_path = os.path.join(images_dir, 'cpath_avg_j')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of individual labor supply distribution averaged
    # over ability j
    npath_avg_s = (npath[:, :, :T2] * lambdas_arr).sum(axis=1)
    cmap_n = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'average lab. supply $n_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat_s, smat, npath_avg_s, rstride=strideval,
                    cstride=strideval, cmap=cmap_n)
    images_path = os.path.join(images_dir, 'npath_avg_s')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of individual labor supply distribution averaged
    # over age s
    npath_avg_j = (npath[:, :, :T2] * pop_dist_age).sum(axis=0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'ability-$j$')
    ax.set_zlabel(r'average lab. supply $n_{j,t}$')
    strideval = max(int(1), int(round(J / 10)))
    ax.plot_surface(tmat_j, jmat, npath_avg_j, rstride=strideval,
                    cstride=strideval, cmap=cmap_n)
    images_path = os.path.join(images_dir, 'npath_avg_j')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of individual savings distribution averaged over
    # ability j
    bpath_avg_s = (bpath[:, :, :T2] * lambdas_arr).sum(axis=1)
    cmap_b = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'average savings $b_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat_s, smat, bpath_avg_s, rstride=strideval,
                    cstride=strideval, cmap=cmap_b)
    images_path = os.path.join(images_dir, 'bpath_avg_s')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot time path of individual savings distribution averaged over
    # age s
    bpath_avg_j = (bpath[:, :, :T2] * pop_dist_age).sum(axis=0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'ability-$j$')
    ax.set_zlabel(r'average savings $b_{j,t}$')
    strideval = max(int(1), int(round(J / 10)))
    ax.plot_surface(tmat_j, jmat, bpath_avg_j, rstride=strideval,
                    cstride=strideval, cmap=cmap_b)
    images_path = os.path.join(images_dir, 'bpath_avg_j')
    plt.savefig(images_path)
    # plt.show()
    plt.close()
