from __future__ import print_function
import numpy as np
from lmfit import Parameters
import os


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
    num_answ = Parameters()
    str_answ = dict()

    for string in fin:
        string = string.replace(',', '')
        if string.startswith('#'):
            continue
        elif len(string) == 0:
            continue

        args = [elem.split('=') for elem in string.split()]
        name, value = args[0]

        if len(args) == 1:
            try:
                num_answ.add(name, value=float(value))
            except ValueError:
                str_answ[name] = value

        else:
            args = dict(args[1:])

            try:
                value = float(value)
            except ValueError:
                raise ValueError('Check the value of parameter in: %s' % string)

            vary, minimum, maximum = True, True, None

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

            num_answ.add(name, value=float(value), vary=vary, min=minimum, max=maximum)
    return num_answ, str_answ


def get_dat(name, normalize=False, bragg=0, template='xy'):
    # TODO: test correct export with different data templates
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
    fin = [elem for elem in fin if elem[0] != '#']    # delete comment strings

    columns = len(fin[0])
    fin = np.array(fin)

    if columns == 1:
        raise ValueError('.dat file must contain at least two columns')

    x = np.array([elem[0]+bragg for elem in fin])
    y = np.array([elem[1] for elem in fin])

    try:
        if template == 'xy':
            if normalize:
                y = y / np.sin(np.deg2rad(x))
            else:
                pass
            answ = x, y

        elif template == 'xyyerr':
            if normalize:
                y = y / np.sin(np.deg2rad(x))
            else:
                pass
            yerror = np.array([elem[2] for elem in fin])
            answ = x, y, yerror

        elif template == 'xyy':
            y1 = np.array([elem[1] for elem in fin])
            y2 = np.array([elem[2] for elem in fin])
            y = y1 + y2 / 2.0
            yerror = np.abs(y - y1)

            if normalize:
                y = y / np.sin(np.deg2rad(x))
            else:
                pass

            answ = x, y, yerror

        elif template == 'xxerryyerr':
            xerror = np.array([elem[2] for elem in fin])
            yerror = np.array([elem[3] for elem in fin])

            if normalize:
                y = y / np.sin(np.deg2rad(x))
            else:
                pass

            answ = x, y, xerror, yerror

        else:
            raise ValueError('Check the "template" parameter')

    except ValueError:
        raise ValueError('Check the "template" parameter')

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
    fin = open(name).read().split('\n')[5:]     # removes first 5 lines, assuming grd file as x-server's output
    fout = open(temp_name, mode='w')  # writes only a table to name_table.grd to prevent overwriting

    print(*fin, sep='\n', file=fout)
    del fout
    table = np.array(np.loadtxt(temp_name))

    table = np.fliplr(table)

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
    hwidth = (width-1)/2
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
        answ[i] = np.average(array[left_border:right_border+1])

    return answ
