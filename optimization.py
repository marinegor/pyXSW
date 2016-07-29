from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import os
from export import get_grd, get_dat


def intensity(table, angles, amplitude, sigma, x0, angle_slope, bragg=0, zmin=0, zmax=100):
    """
    Considers a distribution of atoms as 1 gaussian peak q = amplitude*exp(-(x-x0)**2/2sigma**2).

    Gives I(theta) for a given theta range. If 'bragg' is given, substitutes it from each angle during
    calculations (useful, when experimental 'angles' are adjusted to the bragg angle. Please, check
    Stepanov's server output in reflection calculation or experimental data to find it.

    I(theta) = sum_over_z(gaussian(z)*standing_wave_table(theta, z))

    :param bragg: bragg angle, if not adjusted to it
    :param zmax: upper limit for integration
    :param zmin: lower limit for integration
    :param angle_slope: additional fit parameter for slight amplitude dependence over theta:
           A = amplitude + angle_slope*theta
    :param x0:  position of gaussian peak
    :param sigma: gaussian sigma
    :param amplitude: amplitude of gaussian in atoms distribution nearby the surface
    :param angles: angles range, experimental (usually adjustat to 1st bragg angle)
    :param table: table of intensities from Sergey Stepanov's server calculations
    :rtype: intensity list for the given angle range
    """
    n = len(angles)
    distances = table.shape[1]

    if table.shape[0] != n:
        raise ValueError('Number of points across theta in table %d and angles %d does not match!'%(table.shape[0], n))

    I = np.zeros(n)
    z = np.linspace(zmin, zmax, distances)

    gauss = lambda coord, angle: (amplitude + angle_slope*(angle-bragg))*np.exp(-(coord-x0)**2 / 2.0 / sigma**2)

    gaussian = [[gauss(coord, angle) for coord in z] for angle in angles]
    ibar = gaussian*table

    I = np.array([sum(elem) for elem in ibar])

    return I


def gaussian_residual(params, table, angles, data):
    """
    A residual function for one-gaussian approximation of data
    :param data: experimental data
    :param angles: angles range from experiment
    :param table: table from Stepanov's server
    :param params: must contain 'sigma', 'amplitude' and 'x0' values at least
    :rtype: np.array, shape=(len(angles),)
    """
    try:
        amplitude = params['amplitude']
    except KeyError:
        raise KeyError('Amplitude must be defined!')
    try:
        sigma = params['sigma']
    except KeyError:
        raise KeyError('Sigma must be defined!')
    try:
        x0 = params['x0']
    except KeyError:
        raise KeyError('Sigma must be defined!')
    try:
        angle_slope = params['angle_slope']
    except KeyError:
        angle_slope = 0
    try:
        bragg = params['bragg']
    except KeyError:
        bragg = 0
    try:
        zmin = params['zmin']
    except KeyError:
        zmin = 0
    try:
        zmax = params['zmax']
    except KeyError:
        zmax = 100

    model = intensity(table, angles, amplitude, sigma, x0, angle_slope, bragg, zmin, zmax)

    return abs((data-model))
