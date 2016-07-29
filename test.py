import numpy as np
import matplotlib.pyplot as plt
import os
from export import *

# Dedicated to the testing of functions from export.py

tests_getdat = '/home/errorochka/Dropbox/DESY/pyXSW/tests/get_dat/'
tests_getgrd = '/home/errorochka/Dropbox/DESY/pyXSW/tests/get_grd'


def get_grd_test():
    """
    :rtype: None
    """
    os.chdir(tests_getgrd)
    for fle in os.listdir(tests_getgrd):
        if '~' in fle:
            continue
        print(fle)
        plt.imshow(get_grd(fle), aspect='auto')
        plt.colorbar()
        plt.show()
    return None

def get_dat_test():
    os.chdir(tests_getdat)
    for fle in os.listdir(tests_getdat):
        if '~' in fle:
            continue
        print(fle)
        try:
            x, y = get_dat(fle)
        except:
            print ('Inspect %s'%fle)
        plt.plot(x, y)
    plt.show()
    return None