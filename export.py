from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import os


# Functions dedicated to data export from .dat, .grd and .inp files.


def get_dat(name, normalize=False, bragg=0):
    """
    # Reads the name file and returns a tuple of values in it.
    # Columns must follow one of the following orders:
    # Maximum 1 header string is allowed, 2-4 columns
    # angle   signal
    # angle   signal  signal_error
    # angle   signal  angle_error signal_error
    :param bragg: must be provided, if 'normalize=True'
    :param normalize: 'True', if you want to divide intensity by sin(theta_bragg + theta_exp) for each theta_exp.
    However, for that you have to provide 'bragg' parameter. If it is 0 for your data, please put 1e-9 instead
    of 0 (will not affect calculations, but good for coding).
    :param name: filename
    :return: tuple of values
    """

    if normalize:
        if bragg == 0:
            raise ValueError('Bragg angle must be provided (put 1e-9 instead of 0)')
        else:
            pass
    else:
        bragg = 0

    fin = open(name).read().replace(',', '.').split('\n')  # replace commas by dots to convert to float then
    if fin[-1] == []:
        fin = fin[:-1]

    try:  # check if we have a header
        thereisnoheader = [float(i) for i in fin[0].split()]
    except ValueError:
        thereisnoheader = False

    try:
        if thereisnoheader:
            fin = [[float(i) for i in elem.split()] for elem in fin]
        else:
            fin = [[float(i) for i in elem.split()] for elem in fin[1:]]
    except ValueError:
        raise ValueError('.dat file must contain maximum one header')

    columns = len(fin[0])
    fin = np.array(fin)

    if columns == 1:
        raise ValueError('.dat file must contain at least two columns')

    elif columns == 2:
        x = [elem[0]+bragg for elem in fin if elem]
        if normalize:
            y = [elem[1]/np.sin(np.deg2rad(elem[0]+bragg)) for elem in fin if elem]
        else:
            y = [elem[1] for elem in fin if elem]
        answ = x, y

    elif columns == 3:
        x = [elem[0]+bragg for elem in fin if elem]
        if normalize:
            y = [elem[1]/np.sin(elem[0]+bragg) for elem in fin if elem]
        else:
            y = [elem[1] for elem in fin if elem]
        xerror = [elem[2] for elem in fin if elem]
        answ = x, y, xerror

    elif columns == 4:
        x = [elem[0]+bragg for elem in fin if elem]
        if normalize:
            y = [elem[1]/np.sin(elem[0]+bragg) for elem in fin if elem]
        else:
            y = [elem[1] for elem in fin if elem]
        xerror = [elem[2] for elem in fin if elem]
        yerror = [elem[3] for elem in fin if elem]
        answ = x, y, xerror, yerror

    else:
        raise ValueError('.dat file must contain 2-4 columns')
    return np.array(answ)


def get_grd(name):
    # TODO: try to locate the first data string
    """
    Function simply cuts firts 5 strings of the table and returns an intensity table.
    Adjusted to the data coming from http://x-server.gmca.aps.anl.gov/TER_sl.html
    :param name: filename of .grd file
    :rtype: table[angle][distance] returns an intensity at that coordinates

    """
    if name[-4:] != '.grd':
        raise ValueError('Must be the .grd file')

    temp_name = name[:-4] + '_table.grd'
    fin = open(name).read().split('\n')[5:]
    fout = open(temp_name, mode='w')  # writes only a table to name_table.grd to prevent overwriting

    print(*fin, sep='\n', file=fout)
    del fout
    table = np.array(np.loadtxt(temp_name))
    os.remove(temp_name)

    return table


def get_initials(name):

    """
    Written to simplify input for several experiments on single substrate
    File .prm must contain strings, in each containing parameters written in "name=value"-format.
    Note that wrapping '=' into spaces will lead to a mistake.

    EXAMPLE:
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
    answ = Parameters()

    for string in fin:
        string = string.replace(',', '').lower()
        if string.startswith('#'):
            continue
        elif len(string) == 0:
            continue

        args = [elem.split('=') for elem in string.split()]
        name, value = args[0]

        if len(args) == 1:
            answ.add(name, value=float(value))
        else:
            args = dict(args[1:])

            if 'vary' in args.keys():
                if args['vary'] == 'false':
                    vary = False
                elif args['vary'] == 'true':
                    vary = True
                else:
                    raise ValueError('Check vary= in %s'%string)
            else:
                vary = True

            if 'min' in args.keys():
                try:
                    minimum = float(args['min'])
                except:
                    raise ValueError('Check min= in %s'%string)
            else:
                minimum = None

            if max in args.keys():
                try:
                    maximum = float(args['max'])
                except:
                    raise ValueError('Check max= in %s'%string)
            else:
                maximum = None
            answ.add(name, value=float(value), vary=vary, min=minimum, max=maximum)

    return answ
