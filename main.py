from __future__ import print_function
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
from optimization import *
from export import *
import time
import sys
# from scipy.optimize import rosen, rosen_der, minimize
#
# def callback(x):
#     print ('exact: ', rosen_der(x))
#     # print ('approx: ', _approx_fprime_helper(x, rosen, 1E-8))
#     print ('-----')

# Made up untill lmfit.Parameters.pretty_print() has no file= parameter
orig_stdout = sys.stdout
fout_name = 'tests/'+'pyXSW_' + str(int(time.time()))
fout = open(fout_name + '.log', mode='a')    # log-file, unique for each trial
sys.stdout = fout

print('Output will be written to:\n', file=orig_stdout)
print(fout_name, file=orig_stdout)


initials = str(sys.argv[1])     # reads all fitting parameters from .prm-file
num_params, str_params = get_initials(initials)

# multilayer structure parameters
table = str_params['table']
zmin = float(num_params['zmin'])
num_params.add('zmin', zmin, vary=False)

zmax = float(num_params['zmax'])
num_params.add('zmax', zmax, vary=False)

ml_period = num_params['ml_period']

# experimental data parameters
data = str_params['data']
template = str_params['template']
bragg = num_params['bragg']

# print (params['normalize'])

# fitting parameters
normalize = str_params['normalize']
if normalize == 'true':
    normalize = True
else:
    normalize = False

scan_zmin_zmax = str_params['scan_zmin_zmax']
if scan_zmin_zmax == 'true':
    scan_zmin_zmax = True
else:
    scan_zmin_zmax = False

try:
    window = num_params['window']
except KeyError:
    window = 1

# fitting initial conditions
angle_slope = float(num_params['angle_slope'])
amplitude = float(num_params['amplitude'])
sigma = float(num_params['sigma'])
x0 = float(num_params['x0'])

print('pyXSW is running. '
      'Initial numerical parameters are:\n')
print('pyXSW is running. '
      'Initial numerical parameters are:\n', file=orig_stdout)

num_params.pretty_print()
sys.stdout = orig_stdout
num_params.pretty_print()
sys.stdout = fout

print('\nString parameters are:\n')
print('\nString parameters are:\n', file=orig_stdout)

for key in str_params:
    print(key, str_params[key], sep='=\t')
    print(key, str_params[key], sep='=\t', file=orig_stdout)

theta, real_data, errors = get_dat(data, normalize=normalize, bragg=bragg, template=template)

real_data = smooth(real_data, window)
table_opt = get_grd(table)

# plt.errorbar(theta, real_data, yerr=errors, fmt='ok')
# plt.show()
# exit()

start = time.time()
print('\n\nOptimization is running...\n\n')
print('\n\nOptimization is running...\n\n', file=orig_stdout)

# The most important string -- fitting itself, all the rest is just fun
if scan_zmin_zmax:
    x0s = initial_conditions_list(x0, zmin, zmax, ml_period)
    for value in x0s:

        i = int(value)

        print("""
****************
Model # %d *****
****************
""" % i)

        print("""
****************
Model # %d *****
****************
""" % i, file=orig_stdout)

        num_params.add('x0', value)

        out = minimize(gaussian_residual, num_params, args=(table_opt, theta, real_data, errors))

        Model = intensity(table_opt, theta,
                          out.params['amplitude'],
                          out.params['sigma'],
                          out.params['x0'],
                          out.params['angle_slope'],
                          out.params['bragg'],
                          out.params['zmin'],
                          out.params['zmax'])

        rfactor = sum(abs(Model-real_data))/sum(real_data)

        print('\n\n')
        print('\n\n',file=orig_stdout)

        print('\nFitted parameters are:\n')
        print('\nFitted parameters are:\n', file=orig_stdout)

        out.params.pretty_print()
        sys.stdout = orig_stdout
        out.params.pretty_print()
        sys.stdout = fout

        print('\n')
        print('\n', file=orig_stdout)

        print("\nR-factor=%f" % rfactor)
        print("\nR-factor=%f" % rfactor, file=orig_stdout)

        image_filename = fout_name + '__' + str(i) + '.png'

        plt.plot(theta, real_data, 'ok')
        plt.plot(theta, Model, label='x0=%.2f, sg=%.2f'%(out.params['x0'].value, out.params['sigma'].value))
        plt.legend()
        plt.savefig(image_filename)
        plt.clf()

        print('\nImage is saved into: %s' % image_filename)
        print('\nImage is saved into: %s' % image_filename, file=orig_stdout)

else:

    out = minimize(gaussian_residual, num_params, args=(table_opt, theta, real_data, errors))

    Model = intensity(table_opt, theta,
                      out.params['amplitude'],
                      out.params['sigma'],
                      out.params['x0'],
                      out.params['angle_slope'],
                      out.params['bragg'],
                      out.params['zmin'],
                      out.params['zmax'])

    rfactor = sum(abs(Model-real_data))/sum(real_data)

    print('\n\n')
    print('\n\n', file=orig_stdout)

    print('\nFitted parameters are:\n')
    print('\nFitted parameters are:\n', file=orig_stdout)

    out.params.pretty_print()
    sys.stdout = orig_stdout
    out.params.pretty_print()
    sys.stdout = fout

    print('\n')
    print('\n',file=orig_stdout)

    print("\nR-factor=%f"%rfactor)
    print("\nR-factor=%f"%rfactor, file=orig_stdout)

    image_filename = fout_name + '.png'
    print('\nImage is saved into: %s' % image_filename)
    print('\nImage is saved into: %s' % image_filename, file=orig_stdout)


    plt.plot(theta, real_data, 'ok')
    plt.plot(theta, Model, label='x0=%.2f, sg=%.2f'%(out.params['x0'].value, out.params['sigma'].value))
    plt.legend()
    plt.savefig(image_filename)
    plt.show()


stop = time.time()
print('Total time consumed is %.1f sec\n\n' % (stop - start))
print('Total time consumed is %.1f sec\n\n' % (stop - start), file=orig_stdout)

sys.stdout = orig_stdout
fout.close()

# Our best lattice
# table = '/home/errorochka/Dropbox/DESY/real_data/tr149944_sw.grd'
# Emanuel's best lattice
# table = '/home/errorochka/Dropbox/DESY/real_data/emanuel.grd'
# table = '/home/errorochka/Dropbox/DESY/real_data/emanuel_150.grd'

# theta = '/home/errorochka/Dropbox/DESY/real_data/SGS_lowhum_S.DAT'
# theta = '/home/errorochka/Dropbox/DESY/real_data/SGS_lowhum_K.DAT'
# theta = '/home/errorochka/Dropbox/DESY/real_data/SGS_highhum_S.DAT'
# theta = '/home/errorochka/Dropbox/DESY/real_data/SGS_highhum_K.DAT'
