'''
------------------------------------------------------------------------
This module defines functions for fitting the elliptical disutility of
labor functional form to the constant Frisch elasticity (CFE) disutility
of labor functional form.

This module defines the following functions:
    gen_ellip()
    gen_uprightquad()
    MU_sumsq()
    fit_ellip_CFE()
------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import os

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


def gen_ellip(h, k, a, b, mu, graph=True):
    '''
    --------------------------------------------------------------------
    This plots the general functional form of an ellipse and highlights
    the upper-right quadrant.

    [([x - h] / a) ** mu] + [([y - k] / b) ** mu] = 1
    --------------------------------------------------------------------
    INPUTS:
    h     = scalar, x-coordinate of centroid (h, k)
    k     = scalar, y-coordinate of centroid (h, k)
    a     = scalar > 0, horizontal radius of ellipse
    b     = scalar > 0, vertical radius of ellipse
    mu    = scalar > 0, curvature parameter of ellipse
    graph = boolean, =True if graph output

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    N    = integer > 0, number of elements in the support of x
    xvec = (N,) vector, support of x variable
    yvec = (N,) vector, values of y corresponding to the upper-right
           quadrant values of the ellipse from xvec

    FILES CREATED BY THIS FUNCTION:
        images/EllipseGen.png

    RETURNS: xvec, yvec
    --------------------------------------------------------------------
    '''
    N = 1000
    xvec = np.linspace(h, h + a, N)
    yvec = b * ((1 - (((xvec - h) / a) ** mu)) ** (1 / mu)) + k
    if graph:
        e1 = Ellipse((h, k), 2 * a, 2 * b, 360.0, linewidth=2.0,
                     fill=False, label='Full ellipse')
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(e1)
        plt.plot(xvec, yvec, color='r', linewidth=4,
                 label='Upper-right quadrant')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim((h - 1.6 * a, h + 1.6 * a))
        plt.ylim((k - 1.4 * b, k + 1.4 * b))
        # plt.legend(loc='upper right')
        figname = "images/EllipseGen"
        plt.savefig(figname)
        print("Saved figure: " + figname)
        # plt.show()
        plt.close()

    return xvec, yvec


def gen_uprightquad(hvec, kvec, avec, bvec, muvec, graph=True):
    '''
    --------------------------------------------------------------------
    This plots only the upper-right quadrant of potentially multiple
    ellipses of the form:

    [([x - h] / a) ** mu] + [([y - k] / b) ** mu] = 1
    --------------------------------------------------------------------
    INPUTS:
    hvec  = (I,) vector, x-coordinate of centroid (h, k) of each of I
            ellipses
    kvec  = (I,) vector, y-coordinate of centroid (h, k) of each of I
            ellipses
    avec  = (I,) vector > 0, horizontal radius of ellipse for each of I
            ellipses
    bvec  = (I,) vector > 0, vertical radius of ellipse for each of I
            ellipses
    muvec = (I,) vector > 0, curvature parameter of ellipse for each of
            I ellipses
    graph = boolean, =True if graph output

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    II   = integer >= 1, number of ellipses to produce
    N    = integer >= 1, number of elements (many) in the support of x
    xmat = (I, N) matrix, support of x variable for I ellipses
    ymat = (I, N) matrix, values of y corresponding to the upper-right
           quadrant values of I ellipses from xmat

    FILES CREATED BY THIS FUNCTION:
        images/UpRightQuadEllipses.png

    RETURNS: xmat, ymat
    --------------------------------------------------------------------
    '''
    II = len(hvec)
    N = 1000
    xmat = np.zeros((II, N))
    ymat = np.zeros((II, N))
    for i in range(II):
        xmat[i, :] = np.linspace(hvec[i], hvec[i] + avec[i], N)
        ymat[i, :] = \
            (bvec[i] *
             ((1 - (((xmat[i, :] - hvec[i]) / avec[i]) ** muvec[i])) **
             (1 / muvec[i])) + kvec[i])
    if graph:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(II):
            plt.plot(xmat[i, :], ymat[i, :], linewidth=4,
                     label='Ellipse ' + str(i + 1))
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        # plt.xlim((hvec - 1.6 * avec, hvec + 1.6 * avec))
        # plt.ylim((kvec - 1.4 * bvec, kvec + 1.4 * bvec))
        plt.legend(loc='upper right')
        figname = "images/UpRightQuadEllipses"
        plt.savefig(figname)
        print("Saved figure: " + figname)
        # plt.show()
        plt.close()

    return xmat, ymat


def MU_sumsq(ellip_params, *args):
    '''
    --------------------------------------------------------------------
    This function calculates the sum of squared errors between two
    marginal utility of leisure functions across the support of leisure
    --------------------------------------------------------------------
    INPUTS:
    ellip_params = (2,) vector, scale parameter b and shape parameter
                   upsilon of elliptical disutility of labor
    args         = length 4 tuple,
                   (Frisch, CFE_scale, l_tilde, labor_sup)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    b_ellip    = scalar > 0, scale parameter of elliptical disutility of
                 labor
    upsilon    = scalar > 1, shape parameter of elliptical disutility of
                 labor
    Frisch     = scalar > 0, Frisch elasticity of labor supply in CFE
                 disutility of labor function
    CFE_scale  = scalar > 0, level parameter for CFE disutility of labor
                 function
    l_tilde    = scalar > 0, time endowment for each agent each period
    labor_sup  = (N,) vector, points in support of labor supply
    MU_CFE     = (N,) vector, CFE marginal disutility of labor for
                 labor_sup
    MU_ellip   = (N,) vector, elliptical marginal disutility of labor
                 for labor_sup
    sumsq      = scalar > 0, sum of squared errors between elliptical
                 and CFE marginal utility functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: sumsq
    --------------------------------------------------------------------
    '''
    b_ellip, upsilon = ellip_params
    Frisch, CFE_scale, l_tilde, labor_sup = args

    MU_CFE = CFE_scale * (labor_sup ** (1 / Frisch))
    MU_ellip = ((b_ellip / l_tilde) *
                ((labor_sup / l_tilde) ** (upsilon - 1)) *
                ((1 - ((labor_sup / l_tilde) ** upsilon)) **
                ((1 - upsilon) / upsilon)))
    sumsq = ((MU_ellip - MU_CFE) ** 2).sum()

    return sumsq


def fit_ellip_CFE(ellip_init, cfe_params, l_tilde, graph):
    '''
    --------------------------------------------------------------------
    This function estimates the elliptical disutility of labor
    parameters
    --------------------------------------------------------------------
    INPUTS:
    ellip_init = (2,) vector, initial guesses for b and upsilon
    cfe_params = (2,) vector, parameters for CFE disutility of labor
                 parameters: Frisch elasticity and scale parameter
    l_tilde    = scalar > 0, time endowment to each agent each period
    graph      = Boolean, =True want to save plots of results

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Frisch       = scalar > 0, Frisch elasticity of labor supply
    CFE_scale    = scalar > 0, scale parameter in CFE disutil. of labor
    labor_min    = scalar in (0, labor_max), lower bound of support of
                   labor supply to be used in estimation of parameters
    labor_max    = scalar in (labor_min, l_tilde), upper bound of
                   support of labor supply to be used in estimation of
                   parameters
    labor_N      = integer > 30, number of points in support of labor
                   supply to be used in estimation of parameters
    labor_sup    = (labor_N,) vector, points in support of labor supply
                   to be used in estimation of parameters
    fit_args     = length 3 tuple, (Frisch, CFE_scale, labor_sup)
    bnds_elp     = length 2 tuple, lower and upper bound pairs for b and
                   upsilon elliptical disutility of labor parameters
    ellip_params = length 9 dictionary, output from opt.minimize
    b_ellip      = scalar > 0, scale parameter in elliptical disutility
                   of labor function
    upsilon      = scalar > 1, shape parameter in elliptical disutility
                   of labor function
    sumsq        = scalar > 0, sum of squared errors criterion from
                   opt.minimize

    FILES CREATED BY THIS FUNCTION:
        images/EllipVsCFE_MargUtil.png

    RETURNS: b_ellip, upsilon
    --------------------------------------------------------------------
    '''
    Frisch, CFE_scale = cfe_params
    labor_min = 0.05
    labor_max = 0.95 * l_tilde
    labor_N = 1000
    labor_sup = np.linspace(labor_min, labor_max, labor_N)
    fit_args = (Frisch, CFE_scale, l_tilde, labor_sup)
    bnds_elp = ((1e-12, None), (1 + 1e-12, None))
    ellip_params = opt.minimize(
        MU_sumsq, ellip_init, args=(fit_args), method='L-BFGS-B',
        bounds=bnds_elp)
    b_ellip, upsilon = ellip_params.x
    sumsq = ellip_params.fun
    if ellip_params.success:
        print('SUCCESSFULLY ESTIMATED ELLIPTICAL UTILITY.')
        print('b=', b_ellip, ' upsilon=', upsilon, ' SumSq=', sumsq)

        if graph:
            '''
            ------------------------------------------------------------
            Plot marginal disutilities of labor for CFE and fitted
            elliptical disutility of labor
            ------------------------------------------------------------
            cur_path    = string, path name of current directory
            output_fldr = string, folder in current path to save files
            output_dir  = string, total path of images folder
            output_path = string, path of file name of figure to save
            MU_ellip    = (labor_N,) vector, elliptical marginal
                          utilities for values of labor supply
            MU_CFE      = (labor_N,) vector, CFE marginal utilities for
                          values of labor supply
            ------------------------------------------------------------
            '''
            # Create directory if images folder does not already exist
            cur_path = os.path.split(os.path.abspath(__file__))[0]
            output_fldr = "images"
            output_dir = os.path.join(cur_path, output_fldr)
            if not os.access(output_dir, os.F_OK):
                os.makedirs(output_dir)

            # Plot steady-state consumption and savings distributions
            MU_ellip = \
                ((b_ellip / l_tilde) *
                 ((labor_sup / l_tilde) ** (upsilon - 1)) *
                 ((1 - ((labor_sup / l_tilde) ** upsilon)) **
                 ((1 - upsilon) / upsilon)))
            MU_CFE = CFE_scale * (labor_sup ** (1 / Frisch))
            fig, ax = plt.subplots()
            plt.plot(labor_sup, MU_ellip, label='Elliptical MU')
            plt.plot(labor_sup, MU_CFE, label='CFE MU')
            # for the minor ticks, use no labels; default NullFormatter
            minorLocator = MultipleLocator(1)
            ax.xaxis.set_minor_locator(minorLocator)
            plt.grid(b=True, which='major', color='0.65', linestyle='-')
            plt.title('CFE marginal utility versus fitted elliptical',
                      fontsize=20)
            plt.xlabel(r'Labor supply $n_{s,t}$')
            plt.ylabel(r'Marginal disutility')
            plt.xlim((0, l_tilde))
            # plt.ylim((-1.0, 1.15 * (b_ss.max())))
            plt.legend(loc='upper left')
            output_path = os.path.join(output_dir,
                                       'EllipVsCFE_MargUtil')
            plt.savefig(output_path)
            # plt.show()
            plt.close()
    else:
        print('NOT SUCCESSFUL ESTIMATION OF ELLIPTICAL UTILITY')

    return b_ellip, upsilon
