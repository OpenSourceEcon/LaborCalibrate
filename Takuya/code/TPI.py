'''
------------------------------------------------------------------------
This module contains the functions used to solve the transition path
equilibrium using time path iteration (TPI) for the model with S-period
lived agents and endogenous labor supply from Chapter 4 of the OG
textbook.

This Python module imports the following module(s):
    aggregates.py
    firms.py
    households.py
    utilities.py

This Python module defines the following function(s):
    get_path()
    inner_loop()
    get_TPI()
    create_graphs()
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import aggregates as aggr
import firms
import households as hh
import utilities as utils
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

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
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = (aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) +
                 cc)

    return xpath


def inner_loop(rpath, wpath, args):
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
    args  = length 12 tuple, (S, T2, beta, sigma, l_tilde, b_ellip,
            upsilon, chi_n_vec, bvec1, n_ss, In_Tol, diff)
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.get_n_errors()
        hh.get_b_errors()
    OBJECTS CREATED WITHIN FUNCTION:
    S             = integer in [3,80], number of periods an individual
                    lives
    T2            = integer > S, number of periods until steady state
    beta          = scalar in (0,1), discount factor
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    bvec1         = (S,) vector, initial period savings distribution
    In _tol       = scalar > 0, tolerance level for fsolve's in TPI
    diff          = boolean, =True if want difference version of Euler
                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    cpath         = (S, T2+S-1) matrix, time path of the distribution of
                    consumption
    npath         = (S, T2+S-1) matrix, time path of the distribution of
                    labor supply
    bpath         = (S, T2+S-1) matrix, time path of the distribution of
                    savings
    n_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    labor supply Euler errors
    b_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    savings Euler errors
    bSp1_err_path = (S, T2) matrix, residual last period savings, which
                    should be close to zero in equilibrium. Nonzero
                    elements of matrix should only be in first column
                    and first row
    b_err_params  = length 2 tuple, args to pass into
                    hh.get_b_errors()
    p             = integer in [1, S-1], index representing number of
                    periods remaining in a lifetime, used to solve
                    incomplete lifetimes
    cvec          = (p,) vector, individual lifetime consumption
                    decisions
    nvec          = (p,) vector, individual lifetime labor supply
                    decisions
    bvec          = (p,) vector, individual lifetime savings decisions
    b_Sp1         = scalar, savings in last period for next period.
                    Should be zero in equilibrium
    DiagMaskc     = (p, p) boolean identity matrix
    DiagMaskb     = (p-1, p-1) boolean identity matrix
    n_err_params  = length 5 tuple, args to pass into hh.get_n_errors()
    n_err_vec     = (p,) vector, individual lifetime labor supply Euler
                    errors
    b_err_vec     = (p-1,) vector, individual lifetime savings Euler
                    errors
    t             = integer in [0,T2-1], index of time period (minus 1)
    FILES CREATED BY THIS FUNCTION: None
    RETURNS: cpath, npath, bpath, n_err_path, b_err_path, bSp1_err_path
    --------------------------------------------------------------------
    '''
    (S, T2, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, bvec1,
     n_ss, In_Tol, diff) = args
    cpath = np.zeros((S, T2 + S - 1))
    npath = np.zeros((S, T2 + S - 1))
    bpath = np.append(bvec1.reshape((S, 1)),
                      np.zeros((S, T2 + S - 2)), axis=1)
    n_err_path = np.zeros((S, T2 + S - 1))
    b_err_path = np.zeros((S, T2 + S - 1))
    b_err_args = (beta, sigma, diff)

    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    for p in range(1, S):

        if p == 1:
            # p=1 individual only has an s=S labor supply decision n_S
            n_S1_init = n_ss[-1]
            nS1_args = (wpath[0], sigma, l_tilde, chi_n_vec[-1],
                        b_ellip, upsilon, diff, rpath[0], bvec1[-1],
                        0.0)
            results_nS1 = opt.root(hh.get_n_errors, n_S1_init,
                                   args=(nS1_args), method='lm',
                                   tol=In_Tol)
            n_S1 = results_nS1.x
            npath[-1, 0] = n_S1
            n_err_path[-1, 0] = results_nS1.fun
            cpath[-1, 0] = hh.get_cons(rpath[0], wpath[0], bvec1[-1],
                                       0.0, n_S1)

        else:
            # 1<p<S chooses b_{s+1} and n_s and has incomplete lives
            DiagMaskb = np.eye(p - 1, dtype=bool)
            DiagMaskn = np.eye(p, dtype=bool)
            b_sp1_init = np.diag(bpath[S - p + 1:, :p - 1])
            n_s_init = np.hstack((n_ss[S - p],
                                  np.diag(npath[S - p + 1:, :p - 1])))
            bn_init = np.hstack((b_sp1_init, n_s_init))
            bn_args = (rpath[:p], wpath[:p], bvec1[-p], p, beta, sigma,
                       l_tilde, chi_n_vec[-p:], b_ellip, upsilon, diff)
            results_bn = opt.root(hh.bn_errors, bn_init, args=(bn_args),
                                  tol=In_Tol)
            bvec = results_bn.x[:p - 1]
            nvec = results_bn.x[p - 1:]
            b_s_vec = np.append(bvec1[-p], bvec)
            b_sp1_vec = np.append(bvec, 0.0)
            cvec = hh.get_cons(rpath[:p], wpath[:p], b_s_vec, b_sp1_vec,
                               nvec)
            npath[S - p:, :p] = DiagMaskn * nvec + npath[S - p:, :p]
            bpath[S - p + 1:, 1:p] = (DiagMaskb * bvec +
                                      bpath[S - p + 1:, 1:p])
            cpath[S - p:, :p] = DiagMaskn * cvec + cpath[S - p:, :p]
            n_args = (wpath[:p], sigma, l_tilde, chi_n_vec[-p:],
                      b_ellip, upsilon, diff, cvec)
            n_errors = hh.get_n_errors(nvec, *n_args)
            n_err_path[S - p:, :p] = (DiagMaskn * n_errors +
                                      n_err_path[S - p:, :p])
            b_errors = hh.get_b_errors(cvec, rpath[1:p], *b_err_args)
            b_err_path[S - p + 1:, 1:p] = (DiagMaskb * b_errors +
                                           b_err_path[S - p + 1:, 1:p])

    # Solve the complete remaining lifetime decisions of agents born
    # between period t=1 and t=T2
    for t in range(T2):
        DiagMaskb = np.eye(S - 1, dtype=bool)
        DiagMaskn = np.eye(S, dtype=bool)
        b_sp1_init = np.diag(bpath[1:, t:t + S - 1])
        if t == 0:
            n_s_init = np.hstack((n_ss[0],
                                  np.diag(npath[1:, t:t + S - 1])))
        else:
            n_s_init = np.diag(npath[:, t - 1:t + S - 1])
        bn_init = np.hstack((b_sp1_init, n_s_init))
        bn_args = (rpath[t:t + S], wpath[t:t + S], 0.0, S, beta, sigma,
                   l_tilde, chi_n_vec, b_ellip, upsilon, diff)
        results_bn = opt.root(hh.bn_errors, bn_init, args=(bn_args),
                              tol=In_Tol)
        bvec = results_bn.x[:S - 1]
        nvec = results_bn.x[S - 1:]
        b_s_vec = np.append(0.0, bvec)
        b_sp1_vec = np.append(bvec, 0.0)
        cvec = hh.get_cons(rpath[t:t + S], wpath[t:t + S], b_s_vec,
                           b_sp1_vec, nvec)
        npath[:, t:t + S] = DiagMaskn * nvec + npath[:, t:t + S]
        bpath[1:, t + 1:t + S] = (DiagMaskb * bvec +
                                  bpath[1:, t + 1:t + S])
        cpath[:, t:t + S] = DiagMaskn * cvec + cpath[:, t:t + S]
        n_args = (wpath[t:t + S], sigma, l_tilde, chi_n_vec, b_ellip,
                  upsilon, diff, cvec)
        n_errors = hh.get_n_errors(nvec, *n_args)
        n_err_path[:, t:t + S] = (DiagMaskn * n_errors +
                                  n_err_path[:, t:t + S])
        b_errors = hh.get_b_errors(cvec, rpath[t + 1:t + S], *b_err_args)
        b_err_path[1:, t + 1:t + S] = (DiagMaskb * b_errors +
                                       b_err_path[1:, t + 1:t + S])

    return cpath, npath, bpath, n_err_path, b_err_path


def get_TPI(bvec1, args, graphs):
    '''
    --------------------------------------------------------------------
    Solves for transition path equilibrium using time path iteration
    (TPI)
    --------------------------------------------------------------------
    INPUTS:
    bvec1  = (S,) vector, initial period savings distribution
    args   = length 22 tuple, (S, T1, T2, beta, sigma, l_tilde, b_ellip,
             upsilon, chi_n_vec, A, alpha, delta, r_ss, K_ss, L_ss,
             C_ss, b_ss, n_ss, maxiter, Out_Tol, In_Tol, EulDiff, xi)
    graphs = Boolean, =True if want graphs of TPI objects

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        aggr.get_K()
        get_path()
        firms.get_r()
        firms.get_w()
        inner_loop()
        solve_bn_path()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        utils.print_time()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time    = scalar, current processor time in seconds (float)
    S             = integer in [3,80], number of periods an individual
                    lives
    T1            = integer > S, number of time periods until steady
                    state is assumed to be reached
    T2            = integer > T1, number of time periods after which
                    steady-state is forced in TPI
    beta          = scalar in (0,1), discount factor for model period
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    A             = scalar > 0, total factor productivity parameter in
                    firms' production function
    alpha         = scalar in (0,1), capital share of income
    delta         = scalar in [0,1], per-period capital depreciation rt
    r_ss          = scalar > 0, steady-state aggregate interest rate
    K_ss          = scalar > 0, steady-state aggregate capital stock
    L_ss          = scalar > 0, steady-state aggregate labor
    C_ss          = scalar > 0, steady-state aggregate consumption
    b_ss          = (S,) vector, steady-state savings distribution
                    (b1, b2,... bS)
    n_ss          = (S,) vector, steady-state labor supply distribution
                    (n1, n2,... nS)
    maxiter       = integer >= 1, Maximum number of iterations for TPI
    mindist       = scalar > 0, convergence criterion for TPI
    TPI_tol       = scalar > 0, tolerance level for TPI root finders
    xi            = scalar in (0,1], TPI path updating parameter
    diff          = Boolean, =True if want difference version of Euler
                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    K1            = scalar > 0, initial aggregate capital stock
    K1_cnstr      = Boolean, =True if K1 <= 0
    rpath_init    = (T2+S-1,) vector, initial guess for the time path of
                    interest rates
    iter_TPI      = integer >= 0, current iteration of TPI
    dist          = scalar >= 0, distance measure between initial and
                    new paths
    rw_params     = length 3 tuple, (A, alpha, delta)
    Y_params      = length 2 tuple, (A, alpha)
    cnb_params    = length 11 tuple, args to pass into inner_loop()
    rpath         = (T2+S-1,) vector, time path of the interest rates
    wpath         = (T2+S-1,) vector, time path of the wages
    ind           = (S,) vector, integers from 0 to S
    bn_args       = length 14 tuple, arguments to be passed to
                    solve_bn_path()
    cpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual consumption c_{s,t}
    npath         = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply n_{s,t}
    bpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual savings b_{s,t}
    n_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply Euler errors
    b_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual savings Euler errors. First column and
                    first row are identically zero
    bSp1_err_path = (S, T2) matrix, residual last period savings, which
                    should be close to zero in equilibrium. Nonzero
                    elements of matrix should only be in first column
                    and first row
    Kpath_new     = (T2+S-1,) vector, new path of the aggregate capital
                    stock implied by household and firm optimization
    Kpath_cnstr   = (T2+S-1,) Boolean vector, =True if K_t<=0
    Lpath_new     = (T2+S-1,) vector, new path of the aggregate labor
    rpath_new     = (T2+S-1,) vector, updated time path of interest rate
    wpath_new     = (T2+S-1,) vector, updated time path of the wages
    Ypath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    output (GDP) Y_t
    Cpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    consumption C_t
    RCerrPath     = (T2+S-2,) vector, equilibrium time path of the
                    resource constraint error:
                    Y_t - C_t - K_{t+1} + (1-delta)*K_t
    Kpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    capital stock K_t
    Lpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    labor L_t
    tpi_time      = scalar, time to compute TPI solution (seconds)
    tpi_output    = length 14 dictionary, {cpath, npath, bpath, wpath,
                    rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                    n_err_path, b_err_path, RCerrPath, tpi_time}

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        Ypath.png
        C_aggr_path.png
        wpath.png
        rpath.png
        cpath.png
        npath.png
        bpath.png

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    (S, T1, T2, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A,
        alpha, delta, r_ss, K_ss, L_ss, C_ss, b_ss, n_ss, maxiter,
        Out_Tol, In_Tol, EulDiff, xi) = args
    K1, K1_cnstr = aggr.get_K(bvec1)

    # Create time path for r
    rpath_init = np.zeros(T2 + S - 1)
    rpath_init[:T1] = get_path(r_ss, r_ss, T1, 'quadratic')
    rpath_init[T1:] = r_ss

    iter_TPI = int(0)
    dist = 10.0
    rw_params = (A, alpha, delta)
    Y_params = (A, alpha)
    cnb_args = (S, T2, beta, sigma, l_tilde, b_ellip, upsilon,
                chi_n_vec, bvec1, n_ss, In_Tol, EulDiff)
    while (iter_TPI < maxiter) and (dist >= Out_Tol):
        iter_TPI += 1
        rpath = rpath_init
        wpath = firms.get_w(rpath_init, rw_params)
        cpath, npath, bpath, n_err_path, b_err_path = \
            inner_loop(rpath, wpath, cnb_args)
        Kpath_new = np.zeros(T2 + S - 1)
        Kpath_new[:T2], Kpath_cstr = aggr.get_K(bpath[:, :T2])
        Kpath_new[T2:] = K_ss
        Kpath_cstr = np.append(Kpath_cstr, np.zeros(S - 1, dtype=bool))
        Lpath_new = np.zeros(T2 + S - 1)
        Lpath_new[:T2], Lpath_cstr = aggr.get_L(npath[:, :T2])
        Lpath_new[T2:] = L_ss
        Lpath_cstr = np.append(Lpath_cstr, np.zeros(S - 1, dtype=bool))
        rpath_new = firms.get_r(Kpath_new, Lpath_new, rw_params)
        wpath = firms.get_w(rpath_new, rw_params)
        Ypath = aggr.get_Y(Kpath_new, Lpath_new, Y_params)
        Cpath = np.zeros(T2 + S - 1)
        Cpath[:T2] = aggr.get_C(cpath[:, :T2])
        Cpath[T2:] = C_ss
        RCerrPath = (Ypath[:-1] - Cpath[:-1] - Kpath_new[1:] +
                     (1 - delta) * Kpath_new[:-1])
        # Check the distance of rpath_new
        dist = (np.absolute(rpath_new[:T2] - rpath_init[:T2])).sum()
        print(
            'TPI iter: ', iter_TPI, ', dist: ', "%10.4e" % (dist),
            ', max abs all errs: ', "%10.4e" %
            (np.absolute(np.hstack((b_err_path.max(axis=0),
             n_err_path.max(axis=0)))).max()))
        # The resource constraint does not bind across the transition
        # path until the equilibrium is solved
        rpath_init[:T2] = (xi * rpath_new[:T2] +
                           (1 - xi) * rpath_init[:T2])
    if (iter_TPI == maxiter) and (dist > Out_Tol):
        print('TPI reached maxiter and did not converge.')
    elif (iter_TPI == maxiter) and (dist <= Out_Tol):
        print('TPI converged in the last iteration. ' +
              'Should probably increase maxiter_TPI.')
    Kpath = Kpath_new
    Lpath = Lpath_new

    tpi_time = time.clock() - start_time

    tpi_output = {
        'cpath': cpath, 'npath': npath, 'bpath': bpath, 'wpath': wpath,
        'rpath': rpath, 'Kpath': Kpath, 'Lpath': Lpath, 'Ypath': Ypath,
        'Cpath': Cpath, 'n_err_path': n_err_path,
        'b_err_path': b_err_path, 'RCerrPath': RCerrPath,
        'tpi_time': tpi_time}

    # Print maximum resource constraint error. Only look at resource
    # constraint up to period T2 - 1 because period T2 includes K_{t+1},
    # which was forced to be the steady-state
    print('Max abs. RC error: ', "%10.4e" %
          (np.absolute(RCerrPath[:T2 - 1]).max()))

    # Print TPI computation time
    utils.print_time(tpi_time, 'TPI')

    if graphs:
        graph_args = (S, T2)
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
    args       = length 2 tuple, (S, T2)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    S           =
    T2          =
    Kpath       =
    Lpath       =
    rpath       =
    wpath       =
    Ypath       =
    Cpath       =
    cpath       =
    npath       =
    bpath       =
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    tvec        = (T2+S-1,) vector, time period vector
    tgridT      = (T2,) vector, time period vector from 1 to T2
    sgrid       = (S,) vector, all ages from 1 to S
    tmat        = (S, T2) matrix, time periods for decisions ages
                  (S) and time periods (T2)
    smat        = (S, T2) matrix, ages for all decisions ages (S)
                  and time periods (T2)
    cmap_c      =
    cmap_n      =
    bpath_full  =
    sgrid_b     =
    tmat_b      =
    smat_b      =
    cmap_b      =

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        rpath.png
        wpath.png
        Ypath.png
        C_aggr_path.png
        cpath.png
        npath.png
        bpath.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    S, T2 = args
    Kpath = tpi_output['Kpath']
    Lpath = tpi_output['Lpath']
    rpath = tpi_output['rpath']
    wpath = tpi_output['wpath']
    Ypath = tpi_output['Ypath']
    Cpath = tpi_output['Cpath']
    cpath = tpi_output['cpath']
    npath = tpi_output['npath']
    bpath = tpi_output['bpath']

    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

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
    output_path = os.path.join(output_dir, 'Kpath')
    plt.savefig(output_path)
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
    output_path = os.path.join(output_dir, 'Lpath')
    plt.savefig(output_path)
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
    output_path = os.path.join(output_dir, 'Ypath')
    plt.savefig(output_path)
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
    output_path = os.path.join(output_dir, 'C_aggr_path')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of real wage
    fig, ax = plt.subplots()
    plt.plot(tvec, wpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for real wage w')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Real wage $w_{t}$')
    output_path = os.path.join(output_dir, 'wpath')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of real interest rate
    fig, ax = plt.subplots()
    plt.plot(tvec, rpath, marker='D')
    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Time path for real interest rate r')
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Real interest rate $r_{t}$')
    output_path = os.path.join(output_dir, 'rpath')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of individual consumption distribution
    tgridT = np.linspace(1, T2, T2)
    sgrid = np.linspace(1, S, S)
    tmat, smat = np.meshgrid(tgridT, sgrid)
    cmap_c = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual consumption $c_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, cpath[:, :T2], rstride=strideval,
                    cstride=strideval, cmap=cmap_c)
    output_path = os.path.join(output_dir, 'cpath')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of individual labor supply distribution
    cmap_n = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual labor supply $n_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat, smat, npath[:, :T2], rstride=strideval,
                    cstride=strideval, cmap=cmap_n)
    output_path = os.path.join(output_dir, 'npath')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

    # Plot time path of individual savings distribution
    bpath_full = np.vstack((bpath, np.zeros((1, bpath.shape[1]))))
    sgrid_b = np.linspace(1, S + 1, S + 1)
    tmat_b, smat_b = np.meshgrid(tgridT, sgrid_b)
    cmap_b = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'period-$t$')
    ax.set_ylabel(r'age-$s$')
    ax.set_zlabel(r'individual savings $b_{s,t}$')
    strideval = max(int(1), int(round(S / 10)))
    ax.plot_surface(tmat_b, smat_b, bpath_full[:, :T2],
                    rstride=strideval, cstride=strideval, cmap=cmap_b)
    output_path = os.path.join(output_dir, 'bpath')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
