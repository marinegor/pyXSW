from __future__ import print_function
import numpy as np


def intensity_onegaussian(table, angles, amplitude, sigma, x0, zmax, angle_slope=0, zmin=0, get_ibar = False):
    """
    Considers a distribution of atoms as 1 gaussian peak q = amplitude*exp(-(x-x0)**2/2sigma**2).

    Gives I(theta) for a given theta range.

    I(theta) = sum_over_z(gaussian(z)*standing_wave_table(theta, z))

    :param zmax: upper limit for integration
    :param zmin: lower limit for integration (0 by default)
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
        raise ValueError('Number of points across theta in table %d and angles %d does not match!' %
                         (table.shape[0], n))

    z = np.linspace(zmin, zmax, distances)

    # gauss is a function of the distribution
    gauss = lambda coord, angle: (amplitude + angle_slope*angle)*np.exp(-(coord-x0)**2 / 2.0 / sigma**2)

    # 2D-distribution over theta and z range
    gaussian = [[gauss(coord, angle) for coord in z] for angle in angles]
    # multiplying both distributions -- now ibar is the fluorescing density itself
    ibar = gaussian*table

    # that's the fluorescense yelid curve
    answ = np.array([sum(elem) for elem in ibar])

    if get_ibar:
        return answ, ibar
    else:
        return answ


def residual_onegaussian(params, table, angles, data, errors=None):
    """
    A residual function for one-gaussian approximation of data
    :param errors: optional array including errors of the data
    :param data: np.array() experimental data (use get_dat(filename.dat) to get one)
    :param angles: np.array() angles range from experiment (use get_dat(filename.dat) to get one)
    :param table: np.array() table from Stepanov's server (use get_grd(filename.grd) to get)
    :param params: dictionary; must contain 'sigma', 'amplitude' and 'x0' values at least
    :rtype: np.array()
    """
    try:
        amplitude = params['amp']
    except KeyError:
        raise KeyError('Amp must be defined!')
    try:
        sigma = params['sigma']
    except KeyError:
        raise KeyError('Sigma must be defined!')
    try:
        x0 = params['x0']
    except KeyError:
        raise KeyError('x0 must be defined!')
    try:
        angle_slope = params['angle_slope']
    except KeyError:
        angle_slope = 0
    try:
        zmin = params['zmin']
    except KeyError:
        zmin = 0
    try:
        zmax = params['zmax']
    except KeyError:
        raise KeyError('Zmax must be defined')

    model = intensity_onegaussian(table, angles, amplitude, sigma, x0, zmax, angle_slope, zmin=0)

    if errors is not None:  # if we have errors
        return (data-model)**2 / errors**2     # that must be chi-squared criteria with errors
    else:
        return sum(abs(model - data)) / sum(data)   # optimizing r-factor if there are no errors


def intensity_twogaussians(table, angles,
                           amp1, ratio, sigma1, sigma2, x01, x02,
                           zmax, angle_slope=0, zmin=0, get_ibar=False):
    """
    Considers a distribution of atoms as 2 gaussian peaks:
    q = amp1*exp(-(x-x01)**2/2sigma1**2) + amp2*exp(-(x-x02)**2/2sigma2**2),
    where amp2 = amp1*sigma1/sigma2/ratio, providing opportunity to set
    fixed ratio between gaussians area-under-curve.

    Gives I(theta) for a given theta range.

    I(theta) = sum_over_z(gaussian(z)*standing_wave_table(theta, z))

    :param zmax: upper limit for integration
    :param zmin: lower limit for integration (0 by default)
    :param angle_slope: additional fit parameter for slight amplitude dependence over theta:
           A = amplitude + angle_slope*theta
    :param x01, x02:  position of gaussian peak
    :param sigma1, sigma2: gaussian sigma
    :param amp1: amplitude of gaussian in atoms distribution nearby the surface
    :param ratio: ratio between areas under gaussians, area1/area2.
    :param angles: angles range, experimental (usually adjustat to 1st bragg angle)
    :param table: table of intensities from Sergey Stepanov's server calculations
    :rtype: intensity list for the given angle range
    """

    n = len(angles)
    distances = table.shape[1]

    if table.shape[0] != n:
        raise ValueError('Number of points across theta in table %d and angles %d does not match!' %
                         (table.shape[0], n))

    z = np.linspace(zmin, zmax, distances)

    # first gaussian is just normal
    gauss_first = lambda coord, angle: (amp1 + angle_slope*angle)*np.exp(-(coord-x01)**2 / 2.0 / sigma1**2)

    # amplitude of the second gaussian is adjusted so that:
    # amp1*sigma1 / amp2*sigma2 = ratio,
    # where ratio is a user-defined parameter

    gauss_second= lambda coord, angle: (amp1*sigma1/sigma2/ratio + angle_slope*angle)* \
                                       np.exp(-(coord-x02)**2 / 2.0 / sigma2**2)

    gauss = lambda coord, angle: gauss_first(coord, angle) + gauss_second(coord, angle)

    gaussian = [[gauss(coord, angle) for coord in z] for angle in angles]
    ibar = gaussian*table
    answ = np.array([sum(elem) for elem in ibar])

    if get_ibar:
        return answ, ibar
    else:
        return answ


def residual_twogaussians(params, table, angles, data, errors=None):
    """
    A residual function for one-gaussian approximation of data
    :param errors: optional array including errors of the data
    :param data: np.array() experimental data (use get_dat(filename.dat) to get one)
    :param angles: np.array() angles range from experiment (use get_dat(filename.dat) to get one)
    :param table: np.array() table from Stepanov's server (use get_grd(filename.grd) to get)
    :param params: dictionary; must contain 'sigma', 'amplitude' and 'x0' values at least
    :rtype: np.array, shape=(len(angles),)
    """
    try:
        amp1 = params['amp1']
    except KeyError:
        raise KeyError('Amplitude1 must be defined!')
    try:
        sigma1 = params['sigma1']
    except KeyError:
        raise KeyError('Sigma1 must be defined!')
    try:
        x01 = params['x01']
    except KeyError:
        raise KeyError('x01 must be defined!')
    try:
        sigma2 = params['sigma2']
    except KeyError:
        raise KeyError('Sigma2 must be defined!')
    try:
        x02 = params['x02']
    except KeyError:
        raise KeyError('x02 must be defined!')

    try:
        ratio = params['ratio']
    except KeyError:
        raise KeyError('ratio must be defined!')

    try:
        angle_slope = params['angle_slope']
    except KeyError:
        angle_slope = 0
    try:
        zmin = params['zmin']
    except KeyError:
        raise KeyError('Zmin must be defined!')
    try:
        zmax = params['zmax']
    except KeyError:
        raise KeyError('Zmax must be defined')

    model = intensity_twogaussians(table, angles,
                                   amp1, ratio,
                                   sigma1, sigma2,
                                   x01, x02,
                                   zmax, angle_slope,
                                   zmin)

    if errors is not None:  # if we have errors
        return (data-model)**2 / errors**2     # that must be chi-squared criteria with errors
    else:
        return sum(abs(model - data)) / sum(data)   # optimizing r-factor if there are no errors


def initial_conditions_list(x, xmin, xmax, period):
    """
    Returns largest set of arifmetic progressions with given period, included element and within given limits
    :param period: period of progression
    :param xmax: minimum value
    :param xmin: maximum value
    :param x: float() included element
    :rtype: np.array() with possible conditions
    """
    if xmax < xmin:
        raise ValueError('xmin=%f > xmax=%f' % (xmin, xmax))
    if x < xmin:
        raise ValueError('x=%f < xmin=%f' % (x, xmin))
    if x > xmax:
        raise ValueError('x=%f > xmax=%f' % (x, xmax))

    answ = list()
    while x > xmin:
        x -= period

    x += period

    while x < xmax:
        answ.append(x)
        x += period

    return np.array(answ)
