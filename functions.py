from __future__ import print_function
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
from optimization import *
from export import *
import time
import sys
import os

from export import *
from optimization import *


def boolean_translate(dct, key):
    try:
        answ = str(dct[key])
    except KeyError:
        return True
    if answ.lower() == 'false':
        answ = False
    elif answ.lower() == 'true':
        answ = True
    else:
        raise ValueError('Incorrect %s value for str_param' % key)

    return answ


def propertime():
    def longer(elem):
        if len(str(elem)) == 1:
            return '0' + str(elem)
        else:
            return str(elem)

    year = longer(time.localtime().tm_year)
    mon = longer(time.localtime().tm_mon)
    day = longer(time.localtime().tm_mday)
    hour = longer(time.localtime().tm_hour)
    minute = longer(time.localtime().tm_min)
    sec = longer(time.localtime().tm_sec)
    answ = '%s_%s_%s_%s_%s_%s' % (year, mon, day, hour, minute, sec)
    return answ


def logprint(stream1, stream2, string):
    original = sys.stdout
    sys.stdout = stream1
    print(string)
    sys.stdout = stream2
    print(string)
    sys.stdout = original
    return 0


def intensity_onegaussian(table, angles, amplitude, sigma, x0, zmax, angle_slope=0, zmin=0, get_ibar=False):
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
    gauss = lambda coord, angle: (amplitude) * np.exp(-(coord - x0) ** 2 / 2.0 / sigma ** 2) + angle_slope * angle

    # 2D-distribution over theta and z range
    gaussian = [[gauss(coord, angle) for coord in z] for angle in angles]
    # multiplying both distributions -- now ibar is the fluorescing density itself
    ibar = gaussian * table

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

    # doi:10.1107/S0021889806005073, eq. 6
    if errors is not None:
        chisquared = np.sum((data - model) ** 2 / errors ** 2) / (len(angles) - 4)
    else:
        chisquared = np.sum((data - model) ** 2) / (len(angles) - 4)

    rfactor = sum(abs(model - data) / sum(data))
    print("%f\t%f\t%f\t%f\t%f" % (chisquared, rfactor, x0, sigma, amplitude))

    if errors is not None:  # if we have errors
        return (data - model) ** 2 / errors ** 2 / (len(angles) - 4)  # that must be chi-squared criteria with errors
    else:
        return (data - model) ** 2 / (
        len(angles) - 4)  # that must be chi-square criteria with similar errors (or without)


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
    gauss_first = lambda coord, angle: amp1 * np.exp(-(coord - x01) ** 2 / 2.0 / sigma1 ** 2)

    # amplitude of the second gaussian is adjusted so that:
    # amp1*sigma1 / amp2*sigma2 = ratio,
    # where ratio is a user-defined parameter
    gauss_second = lambda coord, angle: (amp1 * sigma1 / sigma2 / ratio) * \
                                        np.exp(-(coord - x02) ** 2 / 2.0 / sigma2 ** 2) + angle_slope * angle

    gauss = lambda coord, angle: gauss_first(coord, angle) + gauss_second(coord, angle)

    gaussian = [[gauss(coord, angle) for coord in z] for angle in angles]
    ibar = gaussian * table
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

    # doi:10.1107/S0021889806005073, eq. 6

    if errors is not None:
        chisquared = np.sum((data - model) ** 2 / errors ** 2) / (len(angles) - 4)
    else:
        chisquared = np.sum((data - model) ** 2) / (len(angles) - 4)

    rfactor = sum(abs(model - data) / sum(data))
    print("%f\t%f\t%f\t%f\t%f\t%f\t%f" % (chisquared, rfactor, x01, x02, sigma1, sigma2, amp1))

    if errors is not None:  # if we have errors
        return (data - model) ** 2 / errors ** 2 / (len(angles) - 7)  # that must be chi-squared criteria with errors
    else:
        return sum(abs(model - data)) / sum(data)  # optimizing r-factor if there are no errors


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


def intensity_liquid(table, angles,
                     z0, c0, lmbda, const, length,
                     zmax, zmin=0,
                     angle_slope=0, get_ibar=False):
    n = len(angles)
    distances = table.shape[1]

    if table.shape[0] != n:
        raise ValueError('Number of points across theta in table %d and angles %d does not match!' %
                         (table.shape[0], n))

    z = np.linspace(zmin, zmax, distances)

    def rho(z0, c0, lmbda, const, length, angle_slope=0):
        def f(z, angle):
            if z < z0:
                return 0

            elif z0 <= z <= z0 + 4 * lmbda:
                return const + c0 * np.exp(- (z - z0) / lmbda) + angle_slope * angle

            elif z0 + 4 * lmbda < z < z0 + 1 * lmbda + length:
                return const + angle_slope * angle

            else:
                return angle_slope * angle

        answ = lambda z, angle: f(z, angle)
        return answ

    dist = rho(z0, c0, lmbda, const, length, angle_slope)
    distribution = np.array([[dist(coord, angle) for coord in z] for angle in angles])
    ibar = distribution * table
    answ = np.array([sum(elem) for elem in ibar])

    if get_ibar:
        return answ, ibar
    else:
        return answ


def residual_liquid(params, table, angles, data, errors=None):
    try:
        z0 = params['z0']
    except KeyError:
        raise KeyError('Define z0')
    try:
        c0 = params['c0']
    except KeyError:
        raise KeyError('Define c0')
    try:
        lmbda = params['lmbda']
    except KeyError:
        raise KeyError('Define lambda')
    try:
        const = params['const']
    except KeyError:
        raise KeyError('Define const')
    try:
        length = params['length']
    except KeyError:
        raise KeyError['length']
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

    # model = intensity_onegaussian(table, angles, amplitude, sigma, x0, zmax, angle_slope, zmin=0)

    model = intensity_liquid(table, angles, z0, c0, lmbda, const, length, zmax, zmin, angle_slope)

    # doi:10.1107/S0021889806005073, eq. 6
    if errors is not None:
        chisquared = np.sum((data - model) ** 2 / errors ** 2) / (len(angles) - 6)
    else:
        chisquared = np.sum((data - model) ** 2) / (len(angles) - 6)

    rfactor = sum(abs(model - data)) / sum(data)

    # print("%f\t%f\t%f\t%f\t%f" % (chisquared, rfactor, x0, sigma, amplitude))
    print(chisquared, rfactor, z0.value, c0.value, lmbda.value, const.value, length.value, sep='\t')

    if errors is not None:  # if we have errors
        return (data - model) ** 2 / errors ** 2  # that must be chi-squared criteria with errors
    else:
        return (data - model) ** 2 / (len(angles) - 6)  # least-squares without errors


# Functions dedicated to data export from .dat, .grd and .inp files.
def get_initials(name):
    """
    Written to simplify input for several experiments on single substrate
    File .prm must contain strings, in each containing parameters written in "name=value"-format.
    Note that wrapping '=' into spaces will lead to a mistake.

    EXAMPLE:
        normalize=True
        angle_slope=0
        zmin=0
        zmax=100

        bragg=1.080, min=0, vary=False

        amplitude=0.001, min=0
        sigma=22, min=0
        x0=29, min=0, max=100

    WRONG:
        x0= 12
        min =32
        bragg = 10

    :param name: filename of type .prm
    :rtype: returns Parameters() object from input file
    """
    if name[-4:] != '.prm':
        raise ValueError('Must be .prm file')

    fin = open(name).read().split('\n')
    num_answ = Parameters()  # numeric parameters -- lmfit format, will be used for fitting
    str_answ = dict()  # string parameters -- ordinary dictionary

    for string in fin:
        string = string.replace(',', '')
        if string.startswith('#'):
            continue
        elif len(string) == 0:
            continue

        args = [elem.split('=') for elem in string.split()]
        name, value = args[0]

        numeric = True

        try:
            value = float(value)
        except ValueError:
            numeric = False

        if len(args) == 1:

            vary = True
            if name == 'zmin' or name == 'zmax':
                vary = False
            elif name == 'angle_slope':
                vary = True
            if numeric:
                num_answ.add(name, value=value, vary=vary)
            else:
                str_answ[name] = value

        else:
            args = dict(args[1:])
            if numeric:  # numeric parameters have a limited number of options
                vary, minimum, maximum = True, None, None

                # default for minimun and maximum are None -- leads to (-inf) or (inf) durint fitting

                for key in args.keys():
                    if key == 'vary':
                        if args['vary'].lower() == 'false':
                            vary = False
                        elif args['vary'].lower() == 'true':
                            vary = True
                        else:
                            raise ValueError("Check 'vary=' in %s" % string)

                    elif key == 'min':
                        try:
                            minimum = float(args['min'])
                        except:
                            raise ValueError("Check 'min=' in %s" % string)

                    elif key == 'max':
                        try:
                            maximum = float(args['max'])
                        except:
                            raise ValueError("Check 'max=' in %s" % string)

                num_answ.add(name, value=value, vary=vary, min=minimum, max=maximum)
            else:
                raise ValueError("Non-numeric parameter can not have another options.\nCheck:\t%s" % string)

    try:
        a = num_answ['angle_slope']
    except KeyError:
        num_answ.add('angle_slope', value=0, vary=True)

    try:
        a = num_answ['zmin']
    except KeyError:
        num_answ.add('zmin', value=0, vary=False)

    return num_answ, str_answ


def get_dat(name, normalize=False, bragg=0, template='xy'):
    """
    # Reads the name file and returns a tuple of values in it.
    # Columns must follow one of the following orders:
    # Maximum 1 header string is allowed, 2-4 columns
    # angle   signal                            'xy'
    # angle   signal1 signal2                   'xyy'
    # angle   signal  signal_error              'xyyerr'
    # angle   angle_error   signal  signal_error 'xxerryyerr'

    Comment strings start with '#'

    :param template: type of data in columns. Can be 'xy', 'xyyerr', 'xxyy' or 'xyy'
    :param bragg: must be provided, if 'normalize=True'
    :param normalize: 'True', if you want to divide intensity by sin(theta_bragg + theta_exp) for each theta_exp.
    However, for that you have to provide 'bragg' parameter. If it is 0 for your data, please put 1e-9 instead
    of 0 (will not affect calculations, but good for coding).
    :param name: filename
    :return: tuple of values
    """

    fin = open(name).read().replace(',', '.').split('\n')  # replace commas by dots to convert to float then
    fin = [elem for elem in fin if elem.strip() != '']  # delete empty lines
    fin = [elem for elem in fin if elem[0] != '#']  # delete comment strings
    fin = [elem.split() for elem in fin]
    columns = len(fin[0])
    fin = np.array(fin)  # array containing .dat file as numpy array, without comment lines and empty strings
    # NOTE: fin still contains string values, not float

    if columns == 1:
        raise ValueError('.dat file must contain at least two columns')

    # adds bragg angle -- in case if the measurements are relative to that
    x = np.array([float(elem[0]) + bragg for elem in fin])
    y = np.array([float(elem[1]) for elem in fin])

    # if we want to normalize on flux per surface area
    if normalize:
        y = y / np.sin(np.deg2rad(x))

    if template == 'xy':
        answ = x, y

    elif template == 'xyyerror':
        yerror = np.array([float(elem[2]) for elem in fin])
        answ = x, y, yerror

    elif template == 'xyy':
        y1 = np.array([float(elem[1]) for elem in fin])
        y2 = np.array([float(elem[2]) for elem in fin])
        y = y1 + y2 / 2.0
        yerror = np.abs(y - y1)

        answ = x, y, yerror

    elif template == 'xxerroryyerror':
        xerror = np.array([float(elem[2]) for elem in fin])
        yerror = np.array([float(elem[3]) for elem in fin])

        if normalize:
            y = y / np.sin(np.deg2rad(x))
        else:
            pass

        answ = x, y, xerror, yerror

    else:
        raise ValueError('Check the "template" parameter')

    return answ


def get_grd(name, xserver=True):
    """
    Function simply cuts firts 5 strings of the table and returns an intensity table.
    Adjusted to the data coming from http://x-server.gmca.aps.anl.gov/TER_sl.html
    :param xserver: if True, cuts first 5 strings of .grd file (assuming that it's a standart file from x-server)
    :param name: filename of .grd file
    :rtype: table[angle][distance] returns an intensity at that coordinates

    """
    if name[-4:] != '.grd':
        raise ValueError('Must be the .grd file')

    temp_name = name[:-4] + '_temp_table.grd'

    if xserver:
        offcet = 5
    else:
        offcet = 0
    fin = open(name).read().split('\n')[offcet:]  # removes first 5 lines, assuming grd file as x-server's output

    fout = open(temp_name, mode='w')  # writes only a table to name_table.grd to prevent overwriting

    print(*fin, sep='\n', file=fout)
    del fout
    table = np.array(np.loadtxt(temp_name))

    if xserver:
        table = np.fliplr(table)  # now the order within table is "0-100", not "-100-0"

    os.remove(temp_name)

    return table


def smooth(array, width):
    """
    Window-like smoothing, each element is replaced by [-hwidth:hwidth] average.
    :param width: width of the smoothing window. Cauton: if even, will be turned out into width-1
    :param array: np.array() to be smoothed
    :rtype: smoothed np.array()
    """
    array = np.array(array)
    n = len(array)
    hwidth = (width - 1) / 2
    answ = np.zeros(n)

    for i in range(n):
        left_border = i - hwidth
        right_border = i + hwidth

        #         print('i=%d \t left=%d \t right=%d \t array[i]=%d'%(i, left_border, right_border, array[i]))

        if left_border < 0:
            left_border = 0
        if right_border > n - 1:
            right_border = n

        #         print array[left_border:right_border+1]
        answ[i] = np.average(array[left_border:right_border + 1])

    return answ
