'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the OG-ITA overlapping generations model with S-period lived agents,
endogenous labor supply, heterogeneous ability types, demographics,
growth, and bequests.

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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


def get_SS(init_vals, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply using the bisection method in K
    and L for the outer loop
    --------------------------------------------------------------------
    INPUTS:
    init_vals = length 3 tuple, (r_init, c1_init, factor_init)
    args      = length 19 tuple, (J, S, lambdas, emat, beta, sigma,
                l_tilde, b_ellip, upsilon, chi_n_vec_hat, Z, gamma,
                delta, Bsct_Tol, Eul_Tol, xi, maxiter, mean_ydata,
                init_calc)
    graphs    = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_r()
        firms.get_w()
        utils.print_time()
        inner_loop()
        creat_graphs()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time    = scalar > 0, clock time at beginning of program
    r_init        = scalar > -delta, initial guess for steady-state
                    interest rate
    c1_init       = (J,) vector, initial guess for first period
                    consumption by all J ability types
    factor_init   = scalar > 0, initial guess for factor that scales
                    model units to data units
    J             = integer >= 1, number of heterogeneous ability groups
    S             = integer in [3, 80], number of periods an individual
                    lives
    lambdas       = (J,) vector, income percentiles for distribution of
                    ability within each cohort
    emat          = (S, J) matrix, lifetime ability profiles for each
                    lifetime income group j
    beta          = scalar in (0,1), discount factor for each model per
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment of each agent each period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec_hat = (S,) vector, values for chi^n_s hat from data
    Z             = scalar > 0, total factor productivity parameter in
                    firms' production function
    gamma         = scalar in (0,1), capital share of income
    delta         = scalar in [0,1], model-period depreciation rate of
                    capital
    Bsct_Tol      = scalar > 0, tolerance level for outer-loop bisection
                    method
    Eul_Tol       = scalar > 0, tolerance level for inner-loop root
                    finder
    xi            = scalar in (0, 1], SS updating parameter in outer-
                    loop bisection method
    maxiter       = integer >= 1, maximum number of iterations in outer-
                    loop bisection method
    mean_ydata    = scalar > 0, average household income from the data
    init_calc     = boolean, =True if initial steady-state calculation
                    with chi_n_vec = 1 for all s
    iter_SS       = integer >= 0, index of iteration number
    dist          = scalar > 0, distance metric for current iteration
    c1_options    = length 1 dict, options to pass into
                    opt.root(c1_bSp1err,...)
    rw_params     = length 3 tuple, (Z, gamma, delta) args to pass into
                    firms.get_w()
    c1ss_init     = (J,) vector, initial guess for css_{j,1}, updates
                    each iteration
    lambda_mat    = (S, J) matrix, lambdas vector copied down S rows
    w_init        = scalar, initial value for wage
    chi_n_vec     = (S,) vector, chi^n_s values that scale the
                    disutility of labor supply
    inner_args    = length 16 tuple, args to pass into inner_loop()
    K_new         = scalar > 0, updated K given r_init and w_init
    L_new         = scalar > 0, updated L given r_init and w_init
    cmat          = (S, J) matrix, lifetime household consumption by age
                    s and ability j
    nmat          = (S, J) matrix, lifetime household labor supply by
                    age s and ability j
    bmat          = (S, J) matrix, lifetime household savings by age s
                    and ability j (b1, b2,...bS)
    b_Sp1_vec     = (J,) vector, household savings in last period for
                    each ability type, should be arbitrarily close to 0
    r_new         = scalar > 0, updated interest rate given nmat, bmat
    w_new         = scalar > 0, updated wage given nmat and bmat
    n_err_mat     = (S, J) matrix, labor supply Euler errors given
                    r_init and w_init
    b_err_mat     = (S, J) matrix, savings Euler errors given r_init
                    and w_init. First row is identically zeros
    all_errors    = (2JS + J,) vector, (n_errors, b_errors, b_Sp1_vec)
    avg_inc_model = scalar > 0, average household income in model
    factor_new    = scalar > 0, updated factor value given nmat, bmat,
                    r_new, and w_new
    c_ss          = (S, J) matrix, steady-state lifetime consumption
    n_ss          = (S, J) matrix, steady-state lifetime labor supply
    b_ss          = (S, J) matrix, steady-state wealth (savings) at
                    beginning of each period (b1, b2, ...bS)
    b_Sp1_ss      = (J,) vector, steady-state savings in last period of
                    life for all J ability types, approx. 0 in equilbrm
    n_err_ss      = (S, J) matrix, lifetime labor supply Euler errors
    b_err_ss      = (S, J) matrix, lifetime savings Euler errors
    r_ss          = scalar > 0, steady-state interest rate
    w_ss          = scalar > 0, steady-state wage
    K_ss          = scalar > 0, steady-state aggregate capital stock
    L_ss          = scalar > 0, steady-state aggregate labor
    Y_params      = length 2 tuple, (Z, gamma)
    Y_ss          = scalar > 0, steady-state aggregate output (GDP)
    C_ss          = scalar > 0, steady-state aggregate consumption
    RCerr_ss      = scalar, resource constraint error
    ss_time       = scalar, seconds elapsed for steady-state computation
    ss_output     = length 16 dict, steady-state objects {c_ss, n_ss,
                    b_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
                    n_err_ss, b_err_ss, RCerr_ss, ss_time, chi_n_vec,
                    factor_ss}

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    r_init, BQ_init, factor_init, bmat_init, nmat_init = init_vals
    (J, E, S, lambdas, emat, mort_rates, imm_rates, omega_SS, g_n_SS,
        zeta_mat, chi_n_vec_hat, chi_b_vec, beta, sigma, l_tilde,
        b_ellip, upsilon, g_y, Z, gamma, delta, SS_tol_outer,
        SS_tol_inner, xi, maxiter, mean_ydata, init_calc) = args
    iter_SS = 0
    dist = 10
    rw_params = (Z, gamma, delta)
    BQ_args = (lambdas, omega_SS, mort_rates, g_n_SS)
    lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
    omega_mat = np.tile(omega_SS.reshape((S, 1)), (1, J))

    while (iter_SS < maxiter) and (dist >= SS_tol_outer):
        iter_SS += 1
        w_init = firms.get_w(r_init, rw_params)
        if init_calc:
            print('Factor init is one. No modification performed to ' +
                  'chi_n_vec. Dist calculated without factor.')
            chi_n_vec = chi_n_vec_hat
        else:
            chi_n_vec = chi_n_vec_hat * (factor_init ** (sigma - 1))

        rpath = r_init * np.ones(S)
        wpath = w_init * np.ones(S)
        BQpath = BQ_init * np.ones(S)
        cnb_args = (S, lambdas, emat, mort_rates, omega_SS, zeta_mat,
                    chi_n_vec, chi_b_vec, beta, sigma, l_tilde, b_ellip,
                    upsilon, g_y, g_n_SS, SS_tol_inner)
        cmat, nmat, bmat, n_err_mat, b_err_mat = \
            hh.get_cnb_mats(bmat_init, nmat_init, np.zeros(J), rpath,
                            wpath, BQpath, cnb_args)
        K_args = (lambdas, omega_SS, imm_rates, g_n_SS)
        K_new, K_cstr = aggr.get_K(bmat, K_args)
        L_args = (emat, lambdas, omega_SS)
        L_new, L_cstr = aggr.get_L(nmat, L_args)
        r_new = firms.get_r(K_new, L_new, rw_params)
        w_new = firms.get_w(r_new, rw_params)
        BQ_new = aggr.get_BQ(bmat, r_new, BQ_args)

        all_errors = \
            np.hstack((b_err_mat.flatten(), n_err_mat.flatten()))
        if init_calc:
            dist = max(np.absolute((r_new - r_init) / r_init),
                       np.absolute((BQ_new - BQ_init) / BQ_init))
        else:
            avg_inc_model = (omega_mat * lambda_mat *
                             (r_new * bmat + w_new * emat * nmat)).sum()
            factor_new = mean_ydata / avg_inc_model
            print('mean_ydata: ', mean_ydata,
                  ', avg_inc_model: ', avg_inc_model)
            print('r_new: ', r_new, 'BQ_new', BQ_new,
                  'factor_new: ', factor_new)
            dist = max(np.absolute((r_new - r_init) / r_init),
                       np.absolute((BQ_new - BQ_init) / BQ_init),
                       np.absolute((factor_new - factor_init) /
                                   factor_init))
            factor_init = xi * factor_new + (1 - xi) * factor_init

        r_init = xi * r_new + (1 - xi) * r_init
        BQ_init = xi * BQ_new + (1 - xi) * BQ_init
        print('SS Iter=', iter_SS, ', SS Dist=', '%10.4e' % (dist),
              ', Max Abs Err=', '%10.4e' %
              (np.absolute(all_errors).max()))
        bmat_init = bmat
        nmat_init = nmat

    chi_n_vec = chi_n_vec_hat * (factor_init ** (sigma - 1))
    c_ss = cmat.copy()
    n_ss = nmat.copy()
    b_ss = bmat.copy()
    n_err_ss = n_err_mat.copy()
    b_err_ss = b_err_mat.copy()
    r_ss = r_new.copy()
    w_ss = w_new.copy()
    BQ_ss = BQ_new.copy()
    K_ss = K_new.copy()
    L_ss = L_new.copy()
    factor_ss = factor_init
    I_args = (lambdas, omega_SS, imm_rates, g_n_SS, g_y, delta)
    I_ss = aggr.get_I(b_ss, I_args)
    NX_args = (lambdas, omega_SS, imm_rates, g_y)
    NX_ss = aggr.get_NX(b_ss, NX_args)
    Y_params = (Z, gamma)
    Y_ss = aggr.get_Y(K_ss, L_ss, Y_params)
    C_args = (lambdas, omega_SS)
    C_ss = aggr.get_C(c_ss, C_args)
    RCerr_ss = Y_ss - C_ss - I_ss - NX_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'c_ss': c_ss, 'n_ss': n_ss, 'b_ss': b_ss, 'BQ_ss': BQ_ss,
        'w_ss': w_ss, 'r_ss': r_ss, 'K_ss': K_ss, 'L_ss': L_ss,
        'I_ss': I_ss, 'Y_ss': Y_ss, 'C_ss': C_ss, 'NX_ss': NX_ss,
        'n_err_ss': n_err_ss, 'b_err_ss': b_err_ss,
        'RCerr_ss': RCerr_ss, 'ss_time': ss_time,
        'chi_n_vec': chi_n_vec, 'factor_ss': factor_ss}
    print('n_ss is: ', n_ss)
    print('b_ss is: ', b_ss)
    print('c_ss is: ', c_ss)
    print('K_ss=', K_ss, ', L_ss=', L_ss, 'Y_ss=', Y_ss)
    print('r_ss=', r_ss, ', w_ss=', w_ss, 'C_ss=', C_ss)
    print('BQ_ss=', BQ_ss, ', I_ss=', I_ss, ', NX_ss=', NX_ss)
    print('Maximum abs. labor supply Euler error is: ',
          np.absolute(n_err_ss).max())
    print('Maximum abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('Resource constraint error is: ', RCerr_ss)
    print('Final chi_n_vec from SS:', chi_n_vec)
    print('SS factor is: ', factor_ss)

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        gr_args = (E, S, J, chi_n_vec, lambdas)
        create_graphs(c_ss, b_ss, n_ss, gr_args)

    return ss_output


def create_graphs(c_ss, b_ss, n_ss, args):
    '''
    --------------------------------------------------------------------
    Plot steady-state equilibrium results
    --------------------------------------------------------------------
    INPUTS:
    c_ss      = (S, J) matrix, steady-state lifetime consumption
    b_ss      = (S, J) matrix, steady-state lifetime savings
                (b1, b2, ...bS) where b1 = 0
    n_ss      = (S, J) matrix, steady-state lifetime labor supply
    chi_n_vec = (S,) vector, calibrated scale parameters on disutility
                of labor by age
    lambdas   = (J,) vector, income percentiles for distribution of
                ability within each cohort

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    J               = integer >= 1, number of heterogeneous ability
                      groups
    S               = integer in [3, 80], number of periods an
                      individual lives
    cur_path        = string, path name of current directory
    images_fldr     = string, folder in current path to save SS files
    images_dir      = string, total path of SS images folder
    images_cal_fldr = string, folder in current path to save calibration
                      files
    images_cal_dir  = string, total path of calibration images folder
    images_path     = string, path of file name of figure to be saved
    sgrid           = (S,) vector, ages 1,2,...S
    lamcumsum       = (J,) vector, cumulative sum of lambdas percentiles
    jmidgrid        = (J,) vector, midpoints of lamcumsum bins
    smat            = (S, J) matrix, sgrid column vector copied across J
                      columns
    jmat            = (S, J) matrix, jmidgric row vecctor copied down S
                      rows
    cmap_c          = color map for 3D consumption plot
    pct_lb          = scalar in [0, 1], lower bound of percentile bin
    pct_ub          = scalar in [0, 1], upper bound of percentile bin
    this_label      = string, label for percentile bin

    FILES CREATED BY THIS FUNCTION:
        c_ss_3D.png
        c_ss_2D.png
        n_ss_3D.png
        n_ss_2D.png
        b_ss_3D.png
        b_ss_2D.png
        chi_n_vec.png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    E, S, J, chi_n_vec, lambdas = args

    # Create steady-state images directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    images_fldr = 'images/SS'
    images_dir = os.path.join(cur_path, images_fldr)
    if not os.access(images_dir, os.F_OK):
        os.makedirs(images_dir)

    # Create calibration images directory if does not already exist
    images_cal_fldr = 'images/calibration'
    images_cal_dir = os.path.join(cur_path, images_cal_fldr)
    if not os.access(images_cal_dir, os.F_OK):
        os.makedirs(images_cal_dir)

    # Plot 3D steady-state consumption distribution
    sgrid_c = np.arange(E + 1, E + S + 1)
    lamcumsum = lambdas.cumsum()
    jmidgrid = 0.5 * lamcumsum + 0.5 * (lamcumsum - lambdas)
    smat_c, jmat_c = np.meshgrid(sgrid_c, jmidgrid)
    cmap_c = cm.get_cmap('summer')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'ability-$j$')
    ax.set_zlabel(r'indiv. consumption $c_{j,s}$')
    ax.plot_surface(smat_c, jmat_c, c_ss.T, rstride=1,
                    cstride=6, cmap=cmap_c)
    images_path = os.path.join(images_dir, 'c_ss_3D')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot 2D steady-state consumption distribution
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    linestyles = np.array(["-", "--", "-.", ":"])
    markers = np.array(["x", "v", "o", "d", ">", "|"])
    pct_lb = 0
    for j in range(J):
        this_label = (str(int(np.rint(pct_lb))) + " - " +
                      str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                      "%")
        pct_lb += 100 * lambdas[j]
        if j <= 3:
            ax.plot(sgrid_c, c_ss[:, j], label=this_label,
                    linestyle=linestyles[j], color='black')
        elif j > 3:
            ax.plot(sgrid_c, c_ss[:, j], label=this_label,
                    marker=markers[j - 4], color='black')
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'indiv. consumption $c_{j,s}$')
    images_path = os.path.join(images_dir, 'c_ss_2D')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot 3D steady-state labor supply distribution
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'ability-$j$')
    ax.set_zlabel(r'labor supply $n_{j,s}$')
    ax.plot_surface(smat_c, jmat_c, n_ss.T, rstride=1,
                    cstride=6, cmap=cmap_c)
    images_path = os.path.join(images_dir, 'n_ss_3D')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot 2D steady-state labor supply distribution
    minorLocator = MultipleLocator(1)
    fig, ax = plt.subplots()
    linestyles = np.array(["-", "--", "-.", ":"])
    markers = np.array(["x", "v", "o", "d", ">", "|"])
    pct_lb = 0
    for j in range(J):
        this_label = (str(int(np.rint(pct_lb))) + " - " +
                      str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                      "%")
        pct_lb += 100 * lambdas[j]
        if j <= 3:
            ax.plot(sgrid_c, n_ss[:, j], label=this_label,
                    linestyle=linestyles[j], color='black')
        elif j > 3:
            ax.plot(sgrid_c, n_ss[:, j], label=this_label,
                    marker=markers[j - 4], color='black')
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'labor supply $n_{j,s}$')
    images_path = os.path.join(images_dir, 'n_ss_2D')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot 3D steady-state savings/wealth distribution
    sgrid_b = np.arange(E + 1, E + S + 2)
    smat_b, jmat_b = np.meshgrid(sgrid_b, jmidgrid)
    b_ss_w0 = np.vstack((np.zeros(J), b_ss))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'ability-$j$')
    ax.set_zlabel(r'indiv. savings $b_{j,s}$')
    ax.plot_surface(smat_b, jmat_b, b_ss_w0.T, rstride=1,
                    cstride=6, cmap=cmap_c)
    images_path = os.path.join(images_dir, 'b_ss_3D')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot 2D steady-state savings/wealth distribution
    fig, ax = plt.subplots()
    linestyles = np.array(["-", "--", "-.", ":"])
    markers = np.array(["x", "v", "o", "d", ">", "|"])
    pct_lb = 0
    for j in range(J):
        this_label = (str(int(np.rint(pct_lb))) + " - " +
                      str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                      "%")
        pct_lb += 100 * lambdas[j]
        if j <= 3:
            ax.plot(sgrid_b, b_ss_w0[:, j], label=this_label,
                    linestyle=linestyles[j], color='black')
        elif j > 3:
            ax.plot(sgrid_b, b_ss_w0[:, j], label=this_label,
                    marker=markers[j - 4], color='black')
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel(r'age-$s$')
    ax.set_ylabel(r'indiv. savings $b_{j,s}$')
    images_path = os.path.join(images_dir, 'b_ss_2D')
    plt.savefig(images_path)
    # plt.show()
    plt.close()

    # Plot calibrated chi_n_vec
    fig, ax = plt.subplots()
    plt.plot(sgrid_c, chi_n_vec, marker='D',
             label=r'calibrated $\chi^n_s$')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Calibrated disutility of labor scale parameter ' +
    #           'values by age', fontsize=12)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'$\chi^n_s$')
    plt.xlim((E, E + S + 1))
    # plt.ylim((-0.1, 1.15 * (n_ss.max())))
    plt.legend(loc='upper center')
    images_cal_path = os.path.join(images_cal_dir, 'chi_n_vec')
    plt.savefig(images_cal_path)
    # plt.show()
    plt.close()
