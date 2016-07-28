from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

# Functions dedicated to data export from .dat, .grd and .inp files.


def get_dat(name):
    """
    # Reads the name file and returns a tuple of values in it.
    # Columns must follow one of the following orders:
    # Maximum 1 header string is allowed, 2-4 columns
    # angle   signal
    # angle   signal  signal_error
    # angle   signal  angle_error signal_error
    :param name: filename
    :return: tuple of values
    """
    fin = open(name).read().replace(',','.').split('\n')    # replace commas by dots to convert to float then
    if fin[-1] == []:
        fin = fin[:-1]

    try:    # check if we have a header
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
        x = [elem[0] for elem in fin if elem]
        y = [elem[1] for elem in fin if elem]
        return x, y

    elif columns == 3:
        x = [elem[0] for elem in fin if elem]
        y = [elem[1] for elem in fin if elem]
        xerror = [elem[2] for elem in fin if elem]
        return x, y, xerror

    elif columns == 4:
        x = [elem[0] for elem in fin if elem]
        y = [elem[1] for elem in fin if elem]
        xerror = [elem[2] for elem in fin if elem]
        yerror = [elem[3] for elem in fin if elem]
        return x, y, xerror, yerror

    else:
        raise ValueError('.dat file must contain 2-4 columns')


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

    temp_name = name[:-4]+'_table.grd'
    fin = open(name).read().split('\n')[5:]
    fout = open(temp_name, mode='w')    # writes only a table to name_table.grd to prevent overwriting


    print(*fin, sep='\n', file=fout)
    del fout
    table = np.loadtxt(temp_name)
    os.remove(temp_name)

    print (table[0][1])

    return table
