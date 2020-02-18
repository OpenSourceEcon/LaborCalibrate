'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the OG-ITA overlapping generations model with S-
period-lived agents, endogenous labor supply, heterogeneous ability
types demographics, growth, and bequests.

This Python script imports the following module(s):
    demographicsITA.py
    abiility.py
    elliputil.py
    process_json_parameters.py
    SS.py
    households.py
    utilities.py
    TPI.py
    aggregates.py

This Python script calls the following function(s):
    demog.get_pop_objs()
    abil.get_e_interp()
    elp.fit_ellip_CFE()
    pjp.process_calibration_parameters()
    hh.feasible()
    aggr.get_BQ()
    calibrate.get_avg_ydata()
    ss.get_SS()
    calibrate.get_chi_n_vec()
    utils.print_time()
    utils.compare_args()

    aggr.get_K()
    tpi.get_TPI()

Files created by this script:
    OUTPUT/SS/ss_vars_noclb.pkl
    OUTPUT/SS/ss_args_noclb.pkl
    OUTPUT/SS/ss_vars.pkl
    OUTPUT/SS/ss_args.pkl
    OUTPUT/TPI/tpi_vars.pkl
    OUTPUT/TPI/tpi_args.pkl
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import pickle
import os
import time
import demographicsITA as demog
import ability as abil
import elliputil as elp
import SS as ss
import households as hh
import aggregates as aggr
import TPI as tpi
import utilities as utils
import process_json_parameters as pjp

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
E             = integer >= 1, number of periods individuals are not
                economically relevant (youth)
S             = integer in [3,80], number of periods an individual lives
T1            = integer > 1, number of periods after which demographics
                are at their steady state
T2            = integer > T1, number of periods after which endogenous
                model variables are at their steady state
curr_year     = integer >= 2018, current year represented by mode period
                t=1
omega_path    = (T2+S-1, S) matrix, time path for population
                distribution (percents) by economically relevant age
g_n_SS        = scalar, steady-state population growth rate
omega_SS      = (S,) vector, steady-state population distribution
mort_rates    = (S,) vector, mortality rates by age
g_n_path      = (T2+S-1,) vector, time path for population growth rates
imm_rates     = (S,) vector, original immigration rates from data
imm_rates_adj = (S,) vector, adjusted immigration rates that fix pop.
                distribution at t=T1 distribution
omega_preTP   = (S,) vector, population distribution (percent) by age in
                period t=0 (period before t=1)
omega_cur_80  = (80,) vector, current year population distribution
                (percent) by economically relevant age using data ages
imm_rates_mat = (T2+S-1, S) matrix, time path of immigration rates. They
                are original immigration rates until period T1, then
                adjusted immigration rates thereafter
lambdas       = (J,) vector, income percentiles for distribution of
                ability within each cohort
err_msg       = string, error message
J             = integer >= 1, number of heterogeneous ability groups
emat          = (S, J) matrix, lifetime ability profiles for each
                lifetime income group j
zeta_mat      = (S, J) matrix, zeta_{j,s} percent of total bequests that
                goes to each population group (j,s)
beta_annual   = scalar in (0,1), discount factor for one year
beta          = scalar in (0,1), discount factor for each model period
sigma         = scalar > 0, coefficient of relative risk aversion
l_tilde       = scalar > 0, per-period time endowment for every agent
chi_b_vec     = (J,) vector, values for chi^b_j
Z             = scalar > 0, total factor productivity parameter in
                firms' production function
gamma         = scalar in (0,1), capital share of income
delta_annual  = scalar in [0,1], one-year depreciation rate of capital
delta         = scalar in [0,1], model-period depreciation rate of
                capital
g_y_annual    = scalar, annual net productivity growth rate
g_y           = scalar, model period net productivity growth rate
SS_solve      = boolean, =True if want to solve for steady-state
                solution, otherwise retrieve solutions from pickle
SS_tol_outer  = scalar > 0, tolerance level for steady-state outer-loop
                bisection method
SS_tol_inner  = scalar > 0, tolerance level for steady-state inner-loop
                root finders
SS_maxiter    = integer >= 1, maximum iterations of outer-loop steady-
                state bisection method
SS_graphs     = boolean, =True if want graphs of steady-state objects
xi_SS         = scalar in (0, 1], SS updating parameter in outer-loop
                bisection method
TPI_solve     = boolean, =True if want to solve TPI after solving SS
TPI_OutTol    = scalar > 0, tolerance level for outer-loop bisection
                method in TPI
TPI_InTol     = scalar > 0, tolerance level for inner-loop root finder
                in each iteration of TPI
TPI_maxiter   = integer >= 1, Maximum number of iterations for TPI
TPI_graphs    = boolean, =True if want graphs of TPI objects
xi_TPI        = scalar in (0,1], TPI path updating parameter
------------------------------------------------------------------------
'''
# Period parameters and demographics
E = int(20)
S = int(80)
T1 = int(round(3.0 * S))
T2 = int(round(4.0 * S))
curr_year = int(2018)
(omega_path, g_n_SS, omega_SS, mort_rates, g_n_path, imm_rates,
    imm_rates_adj, omega_preTP, omega_cur_80) = \
    demog.get_pop_objs(E, S, T1, T2, curr_year)
imm_rates_mat = \
    np.vstack((np.tile(np.reshape(imm_rates, (1, S)), (T1, 1)),
               np.tile(np.reshape(imm_rates_adj, (1, S)),
                       (T2 + S - 1 - T1, 1))))

# Household utility and ability parameters
lambdas = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
# Make sure that lambdas vector sums to 1.0
if not np.isclose(1.0, lambdas.sum()):
    err_msg = ('ERROR: lambdas vector does not sum to one.')
    raise RuntimeError(err_msg)
J = len(lambdas)
emat = abil.get_e_interp(S, omega_path[0, :], omega_cur_80, lambdas,
                         plot=False)
zeta_mat = (np.tile(omega_SS.reshape((S, 1)), (1, J)) *
            np.tile(lambdas.reshape((1, J)), (S, 1)))
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.5
l_tilde = 1.0
chi_b_vec = 1.0 * np.ones(J)

# Firm parameters
Z = 1.0
gamma = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
g_y_annual = 0.03
g_y = np.exp(g_y_annual * 80 / S) - 1

# SS parameters
SS_solve = False
SS_tol_outer = 1e-8
SS_tol_inner = 1e-13
SS_maxiter = 400
SS_graphs = True
SS_LaborCalib = True
xi_SS = 0.3

# TPI parameters
TPI_solve = True
TPI_OutTol = 1e-8
TPI_InTol = 1e-13
TPI_maxiter = 200
TPI_graphs = True
xi_TPI = 0.05

'''
------------------------------------------------------------------------
Fit elliptical utility function to constant Frisch elasticity (CFE)
disutility of labor function by matching marginal utilities along the
support of leisure
------------------------------------------------------------------------
ellip_graph  = Boolean, =True if want to save plot of fit
b_ellip_init = scalar > 0, initial guess for b
upsilon_init = scalar > 1, initial guess for upsilon
ellip_init   = (2,) vector, initial guesses for b and upsilon
Frisch_elast = scalar > 0, Frisch elasticity of labor supply for CFE
               disutility of labor
CFE_scale    = scalar > 0, scale parameter for CFE disutility of labor
cfe_params   = (2,) vector, values for (Frisch, CFE_scale)
b_ellip      = scalar > 0, fitted value of b for elliptical disutility
               of labor
upsilon      = scalar > 1, fitted value of upsilon for elliptical
               disutility of labor
------------------------------------------------------------------------
'''
ellip_graph = False
b_ellip_init = 1.0
upsilon_init = 2.0
ellip_init = np.array([b_ellip_init, upsilon_init])
Frisch_elast = 0.9
CFE_scale = 1.0
cfe_params = np.array([Frisch_elast, CFE_scale])
b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde,
                                     ellip_graph)

'''
------------------------------------------------------------------------
Solve for the steady-state solution
------------------------------------------------------------------------
cur_path         = string, current file path of this script
ss_output_fldr   = string, cur_path extension of SS output folder path
ss_output_dir    = string, full path name of SS output folder
ss_outfile_noclb = string, path name of file for SS output objects
ss_prmfile_noclb = string, path name of file for SS parameter objects
ss_outputfile    = string, path name of file for SS output objects
ss_paramsfile    = string, path name of file for SS parameter objects
tot_ss_strt_time = scalar, current time on computer
chi_params       = length 7 tuple, arguments to pass in to
                   pjp.process_calibration_parameters()
calibrate        = Calibrate.Calibrate class object
init_chi1        = length 3 dict, precomputed starting values for r,
                   bmat, and nmat that are close to the steady-state
rss_init         = scalar > -delta, initial guess for r_ss
bmat_init        = (S, J) matrix, initial guess for bmat (bj2,...bjS+1)
nmat_init        = (S, J) matrix, initial guess for nmat (nj1,...njS)
rss_path         = (S,) vector, trivial time path of steady-state
                   interest rate
f_args           = length 11 tuple, arguments to pass in to
                   hh.feasible()
b_feas           = (S, J) boolean matrix, =True if bmat > 0
n_feas           = (S, J) boolean matrix, =True if 0 < nmat < l_tilde
c_feas           = (S, J) boolean matrix, =True if cmat > 0
BQ_args          = length 4 tuple, arguments to pass into aggr.get_BQ()
BQ_init          = scalar > 0, initial guess for total bequests
factor_init      = scalar > 0, factor to translate data units to model
                   units
init_calc        = boolean, =True if initial steady-state calculation
                   with chi_n_vec = 1 for all s
mean_ydata       = scalar > 0, average household income from the data
chi_n_vec        = (S,) vector, chi^n_s values for model that scale the
                   disutility of labor supply
ss_init_vals     = length 5 tuple, initial guesses for r_ss, BQ_ss,
                   factor, b_ss, and n_ss
ss_args_noclb    = length 27 tuple, arguments to pass in to ss.get_SS()
                   for case with no calibration chi_n_vec = 1
ss_output_noclb  = length 18 dict, steady-state objects {c_ss, n_ss,
                   b_ss, BQ_ss, w_ss, r_ss, K_ss, L_ss, I_ss, Y_ss,
                   C_ss, NX_ss, n_err_ss, b_err_ss, RCerr_ss, ss_time,
                   chi_n_vec, factor_ss} for case with no calibration
                   chi_n_vec = 1
lambda_mat       = (S, J) matrix, lambdas row vector copied down S rows
omega_mat        = (S, J) matrix, omega_SS column vector copied across J
                   columns
avg_inc_model    = scalar > 0, average income from the model
chi_n_vec_hat    = (S,) vector, chi^n_s hat values from calibration
increments       = integer >= 1, number of increments between chi_n_vec
                   implied by first solution and initial chi_n_vec
convex_comb      = (increments,) vector, percents between 1/incr and 1
chi_n_beg        = (S,) vector, initial chi_n_vec from SS solution
chi_n_end        = (S,) vector, chi_n_vec implied by factor_init and
                   chi_n_vec_hat
incr             = integer>=0, index of increment percentage
ss_args          = length 27 tuple, arguments to pass in to ss.get_SS()
                   for case with calibration
ss_output        = length 18 dict, steady-state objects {c_ss, n_ss,
                   b_ss, BQ_ss, w_ss, r_ss, K_ss, L_ss, I_ss, Y_ss,
                   C_ss, NX_ss, n_err_ss, b_err_ss, RCerr_ss, ss_time,
                   chi_n_vec, factor_ss} for case with calibration
tot_ss_time      = scalar > 0, total time in seconds of SS computation
ss_vars_exst     = boolean, =True if ss_vars.pkl exists
ss_args_exst     = boolean, =True if ss_args.pkl exists
err_msg          = string, error message
ss_vars          = length 27 tuple, saved steady-state objects
cur_ss_args      = length 27 tuple, arguments to pass in to ss.get_SS()
                   for case with calibration
args_same        = boolean, =True if ss_args == cur_ss_args
------------------------------------------------------------------------
'''
# Create OUTPUT/SS directory if does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_output_fldr = 'OUTPUT/SS'
ss_output_dir = os.path.join(cur_path, ss_output_fldr)
if not os.access(ss_output_dir, os.F_OK):
    os.makedirs(ss_output_dir)
ss_outfile_noclb = os.path.join(ss_output_dir, 'ss_vars_noclb.pkl')
ss_prmfile_noclb = os.path.join(ss_output_dir, 'ss_args_noclb.pkl')
ss_outputfile = os.path.join(ss_output_dir, 'ss_vars.pkl')
ss_paramsfile = os.path.join(ss_output_dir, 'ss_args.pkl')

if SS_solve:
    print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION WITH CHI_N_VEC=1')
    tot_ss_strt_time = time.clock()
    # Set initial values
    chi_params = (sigma, l_tilde, ellip_graph, b_ellip_init,
                  upsilon_init, Frisch_elast, CFE_scale)
    calibrate = pjp.process_calibration_parameters()
    init_chi1 = pickle.load(open('init_chi1.pkl', 'rb'))
    # rss_init = 0.15
    rss_init = init_chi1['r_ss']
    # bmat_init = 0.01 * np.ones((S, J))
    bmat_init = init_chi1['b_ss']
    # nmat_init = 0.5 * l_tilde * np.ones((S, J))
    nmat_init = init_chi1['n_ss']
    rss_path = rss_init * np.ones(S)
    f_args = (lambdas, emat, omega_SS, mort_rates, zeta_mat, g_y,
              l_tilde, g_n_SS, Z, gamma, delta)
    b_feas, n_feas, c_feas = hh.feasible(bmat_init, nmat_init, rss_path,
                                         f_args)
    BQ_args = (lambdas, omega_SS, mort_rates, g_n_SS)
    BQ_init = aggr.get_BQ(bmat_init, rss_init, BQ_args)
    factor_init = 1.0
    init_calc = True
    mean_ydata = calibrate.get_avg_ydata()

    chi_n_vec = 1.0 * np.ones(S)
    ss_init_vals = (rss_init, BQ_init, factor_init, bmat_init,
                    nmat_init)
    ss_args_noclb = (J, E, S, lambdas, emat, mort_rates, imm_rates_adj,
                     omega_SS, g_n_SS, zeta_mat, chi_n_vec, chi_b_vec,
                     beta, sigma, l_tilde, b_ellip, upsilon, g_y, Z,
                     gamma, delta, SS_tol_outer, SS_tol_inner, xi_SS,
                     SS_maxiter, mean_ydata, init_calc)
    # print(ss_args_noclb)
    ss_output_noclb = ss.get_SS(ss_init_vals, ss_args_noclb, False)
    # Save ss_output as pickle
    pickle.dump(ss_output_noclb, open(ss_outfile_noclb, 'wb'))
    pickle.dump(ss_args_noclb, open(ss_prmfile_noclb, 'wb'))

    # Update initial guess for factor to factor from chi_n_vec = 1 sol'n
    lambda_mat = np.tile(lambdas.reshape((1, J)), (S, 1))
    omega_mat = np.tile(omega_SS.reshape((S, 1)), (1, J))
    avg_inc_model = \
        (omega_mat * lambda_mat * (ss_output_noclb['r_ss'] *
                                   ss_output_noclb['b_ss'] +
                                   ss_output_noclb['w_ss'] * emat *
                                   ss_output_noclb['n_ss'])).sum()
    print('average_income_model: ', avg_inc_model)
    factor_init = mean_ydata / avg_inc_model
    print('factor_init: ', factor_init)
    chi_n_vec_hat = calibrate.get_chi_n_vec(chi_params)

    # Incrementally solve for larger chi_n_vec vectors until have
    # solutions close to starting solutions with calibration
    increments = int(10)
    convex_comb = np.linspace(1.0 / increments, 1.0, increments)
    chi_n_beg = np.ones(S)
    chi_n_end = factor_init * chi_n_vec_hat
    for incr in range(increments):
        print('Starting increment ', incr + 1, '.')
        init_calc = True
        rss_init = ss_output_noclb['r_ss']
        rss_path = rss_init * np.ones(S)
        BQ_init = ss_output_noclb['BQ_ss']
        bmat_init = ss_output_noclb['b_ss']
        nmat_init = ss_output_noclb['n_ss']
        b_feas, n_feas, c_feas = hh.feasible(bmat_init, nmat_init,
                                             rss_path, f_args)
        ss_init_vals = (rss_init, BQ_init, 1.0, bmat_init, nmat_init)
        chi_n_vec = (convex_comb[incr] * chi_n_end +
                     (1 - convex_comb[incr]) * chi_n_beg)
        ss_args_noclb = \
            (J, E, S, lambdas, emat, mort_rates, imm_rates_adj,
             omega_SS, g_n_SS, zeta_mat, chi_n_vec, chi_b_vec, beta,
             sigma, l_tilde, b_ellip, upsilon, g_y, Z, gamma, delta,
             SS_tol_outer, SS_tol_inner, xi_SS, SS_maxiter, mean_ydata,
             init_calc)
        ss_output_noclb = ss.get_SS(ss_init_vals, ss_args_noclb,
                                    SS_graphs)
        # Save ss_output as pickle
        pickle.dump(ss_output_noclb, open(ss_outfile_noclb, 'wb'))
        pickle.dump(ss_args_noclb, open(ss_prmfile_noclb, 'wb'))

    # Compute steady-state solution and calibrate chi_n_vec
    init_calc = False
    rss_init = ss_output_noclb['r_ss']
    rss_path = rss_init * np.ones(S)
    BQ_init = ss_output_noclb['BQ_ss']
    bmat_init = ss_output_noclb['b_ss']
    nmat_init = ss_output_noclb['n_ss']
    b_feas, n_feas, c_feas = hh.feasible(bmat_init, nmat_init, rss_path,
                                         f_args)
    ss_init_vals = (rss_init, BQ_init, factor_init, bmat_init,
                    nmat_init)
    chi_n_vec = chi_n_vec_hat
    ss_args = (J, E, S, lambdas, emat, mort_rates, imm_rates_adj,
               omega_SS, g_n_SS, zeta_mat, chi_n_vec, chi_b_vec, beta,
               sigma, l_tilde, b_ellip, upsilon, g_y, Z, gamma, delta,
               SS_tol_outer, SS_tol_inner, xi_SS, SS_maxiter,
               mean_ydata, init_calc)

    print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION WITH CALIBRATION')
    print('Solving SS outer loop using bisection method on r, BQ, ' +
          'and factor.')
    ss_output = ss.get_SS(ss_init_vals, ss_args, SS_graphs)

    # Update ss_args with new chi_n_vec
    chi_n_vec = ss_output['chi_n_vec']
    ss_args = (J, E, S, lambdas, emat, mort_rates, imm_rates_adj,
               omega_SS, g_n_SS, zeta_mat, chi_n_vec, chi_b_vec, beta,
               sigma, l_tilde, b_ellip, upsilon, g_y, Z, gamma, delta,
               SS_tol_outer, SS_tol_inner, xi_SS, SS_maxiter,
               mean_ydata, init_calc)

    tot_ss_time = time.clock() - tot_ss_strt_time
    utils.print_time(tot_ss_time, 'Total steady-state')

    # Save ss_output as pickle
    pickle.dump(ss_output, open(ss_outputfile, 'wb'))
    pickle.dump(ss_args, open(ss_paramsfile, 'wb'))

# Don't compute steady-state, get it from pickle
else:
    # Make sure that the SS output files exist
    ss_vars_exst = os.path.exists(ss_outputfile)
    ss_args_exst = os.path.exists(ss_paramsfile)
    if (not ss_vars_exst) or (not ss_args_exst):
        # If the files don't exist, stop the program and run the steady-
        # state solution first
        err_msg = ('ERROR: The SS output files do not exist and ' +
                   'SS_solve=False. Must set SS_solve=True and ' +
                   'compute steady-state solution.')
        raise RuntimeError(err_msg)
    else:
        # If the files do exist, make sure that none of the parameters
        # changed from the parameters used in the solution for the saved
        # steady-state pickle
        ss_args = pickle.load(open(ss_paramsfile, 'rb'))
        ss_vars = pickle.load(open(ss_outputfile, 'rb'))
        chi_n_vec = ss_vars['chi_n_vec']
        mean_ydata = ss_args[25]
        init_calc = False
        cur_ss_args = \
            (J, E, S, lambdas, emat, mort_rates, imm_rates_adj,
             omega_SS, g_n_SS, zeta_mat, chi_n_vec, chi_b_vec, beta,
             sigma, l_tilde, b_ellip, upsilon, g_y, Z, gamma, delta,
             SS_tol_outer, SS_tol_inner, xi_SS, SS_maxiter, mean_ydata,
             init_calc)
        args_same = utils.compare_args(ss_args, cur_ss_args)
        if args_same:
            # If none of the parameters changed, use saved pickle
            print('RETRIEVE STEADY-STATE SOLUTIONS FROM FILE')
            ss_output = pickle.load(open(ss_outputfile, 'rb'))
        else:
            # If any of the parameters changed, end the program and
            # compute the steady-state solution
            err_msg = ('ERROR: Current ss_args are not equal to the ' +
                       'ss_args that produced ss_output. Must solve ' +
                       'for SS before solving transition path. Set ' +
                       'SS_solve=True.')
            raise RuntimeError(err_msg)

'''
------------------------------------------------------------------------
Solve for the transition path equilibrium by time path iteration (TPI)
------------------------------------------------------------------------
tpi_output_fldr = string, cur_path extension of TPI output folder path
tpi_output_dir  = string, full path name of TPI output folder
tpi_outputfile  = string, path name of file for TPI output objects
tpi_paramsfile  = string, path name of file for TPI parameter objects
r_ss            = scalar > 0, steady-state aggregate interest rate
K_ss            = scalar > 0, steady-state aggregate capital stock
L_ss            = scalar > 0, steady-state aggregate labor
C_ss            = scalar > 0, steady-state aggregate consumption
b_ss            = (S, J) matrix, steady-state household savings
                  distribution (b_j1, b_j2,... b_jS)
n_ss            = (S, J) matrix, steady-state household labor supply
                  distribution (n_j1, n_j2,... n_jS)
ub_factor       = scalar > 0, percent above the steady-state of initial
                  wealth of the oldest households
lb_factor       = scalar > 0, percent above the steady-state of initial
                  wealth of the youngest households
init_wgts_vec   = (S,) vector, linear weights by age from lb_factor to
                  ub_factor
init_wgts       = (S, J) matrix, init_wgts_vec column vector copied
                  across J columns
bmat1           = (S, J) matrix, initial period household savings dist'n
K1              = scalar, initial period aggregate capital stock
K1_cstr         = boolean, =True if K1 <= 0
tpi_params      = length 25 tuple, args to pass into tpi.get_TPI()
tpi_output      = length 13 dict, {cpath, npath, bpath, wpath, rpath,
                  Kpath, Lpath, Ypath, Cpath, n_err_path, b_err_path,
                  RCerrPath, tpi_time}
tpi_args        = length 26 tuple, args that were passed in to get_TPI()
------------------------------------------------------------------------
'''
if TPI_solve:
    print('BEGIN EQUILIBRIUM TRANSITION PATH COMPUTATION')

    # Create OUTPUT/TPI directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    tpi_output_fldr = 'OUTPUT/TPI'
    tpi_output_dir = os.path.join(cur_path, tpi_output_fldr)
    if not os.access(tpi_output_dir, os.F_OK):
        os.makedirs(tpi_output_dir)
    tpi_outputfile = os.path.join(tpi_output_dir, 'tpi_vars.pkl')
    tpi_paramsfile = os.path.join(tpi_output_dir, 'tpi_args.pkl')

    r_ss = ss_output['r_ss']
    BQ_ss = ss_output['BQ_ss']
    K_ss = ss_output['K_ss']
    L_ss = ss_output['L_ss']
    NX_ss = ss_output['NX_ss']
    C_ss = ss_output['C_ss']
    b_ss = ss_output['b_ss']
    n_ss = ss_output['n_ss']
    chi_n_vec = ss_output['chi_n_vec']

    # Choose initial period distribution of wealth (bvec1), which
    # determines initial period aggregate capital stock
    ub_factor = 1.04
    lb_factor = 0.98
    init_wgts_vec = ((ub_factor - lb_factor) / (S - 1) *
                     (np.linspace(1, S, S) - 1) + lb_factor)
    init_wgts = np.tile(init_wgts_vec.reshape((S, 1)), (1, J))
    bmat1 = init_wgts * b_ss

    # Make sure init. period distribution is feasible in terms of K
    K_args = (lambdas, omega_preTP, imm_rates, g_n_path[0])
    K1, K1_cstr = aggr.get_K(bmat1, K_args)

    # If initial bvec1 is not feasible end program
    if K1_cstr:
        print('Initial savings distribution is not feasible because ' +
              'K1<=0. Some element(s) of bmat1 must increase.')
    else:
        tpi_params = (J, E, S, T1, T2, lambdas, emat, mort_rates,
                      imm_rates_mat, omega_path, omega_preTP, g_n_path,
                      zeta_mat, chi_n_vec, chi_b_vec, beta, sigma,
                      l_tilde, b_ellip, upsilon, g_y, Z, gamma, delta,
                      r_ss, BQ_ss, K_ss, K1, L_ss, C_ss, NX_ss, b_ss,
                      n_ss, TPI_maxiter, TPI_OutTol, TPI_InTol, xi_TPI)
        tpi_output = tpi.get_TPI(bmat1, tpi_params, TPI_graphs)

        tpi_args = (J, E, S, T1, T2, lambdas, emat, mort_rates,
                    imm_rates_mat, omega_path, g_n_path, zeta_mat,
                    chi_n_vec, chi_b_vec, beta, sigma, l_tilde, b_ellip,
                    upsilon, g_y, Z, gamma, delta, r_ss, BQ_ss, K_ss,
                    K1, L_ss, C_ss, NX_ss, b_ss, n_ss, TPI_maxiter,
                    TPI_OutTol, TPI_InTol, xi_TPI, bmat1)

        # Save tpi_output as pickle
        pickle.dump(tpi_output, open(tpi_outputfile, 'wb'))
        pickle.dump(tpi_args, open(tpi_paramsfile, 'wb'))
