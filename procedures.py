from __future__ import print_function
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
from optimization import *
from export import *
import time
import sys

# TODO: make up a custom print-function that duplicates string into standart output and file


def onegauss_fit(prm_name):
    orig_stdout = sys.stdout
    fout_name = 'tests/'+'pyXSW_' + str(int(time.time()))
    fout = open(fout_name + '.log', mode='a')    # log-file, unique for each trial
    sys.stdout = fout

    print('Output will be written to:\n', file=orig_stdout)
    print(fout_name, file=orig_stdout)

    initials = prm_name    # reads all fitting parameters from .prm-file
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
    bragg = float(num_params['bragg'])

    # print (params['normalize'])

    # fitting parameters
    normalize = str_params['normalize']
    if normalize == 'true':
        normalize = True
    else:
        normalize = False

    scan_zmin_zmax = str_params['scan_zmin_zmax']
    if scan_zmin_zmax.lower() == 'true':
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

    # TODO: organise this part depending on 'template' keyword
    # theta, real_data = get_dat(data, normalize=normalize, bragg=bragg, template=template)
    # errors = np.ones(len(theta))
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

            out = minimize(residual_onegaussian, num_params, args=(table_opt, theta, real_data, errors))

            Model = intensity_onegaussian(table_opt, theta,
                              out.params['amplitude'],
                              out.params['sigma'],
                              out.params['x0'],
                              out.params['zmax'],
                              out.params['angle_slope'],
                              out.params['bragg'],
                              out.params['zmin'],
                              )

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

            chisquared = residual_onegaussian(out, table_opt, theta, real_data, errors)
            print("\nChi-squared=%f" % chisquared)
            print("\nChi-squared=%f" % chisquared, file=orig_stdout)
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
        out = minimize(residual_onegaussian, num_params, args=(table_opt, theta, real_data, errors))

        Model = intensity_onegaussian(table_opt, theta,
                          out.params['amplitude'],
                          out.params['sigma'],
                          out.params['x0'],
                          out.params['zmax'],
                          out.params['angle_slope'],
                          out.params['zmin'],
                          )

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

        chisquared = 1.0 / (len(theta) - 4) * sum((Model - real_data)**2 / errors**2)
        print("\nChi-squared=%f" % chisquared)
        print("\nChi-squared=%f" % chisquared, file=orig_stdout)
        print("\nR-factor=%f"%rfactor)
        print("\nR-factor=%f"%rfactor, file=orig_stdout)

        image_filename = fout_name + '.png'
        print('\nImage is saved into: %s' % image_filename)
        print('\nImage is saved into: %s' % image_filename, file=orig_stdout)

        plt.plot(theta, real_data, 'ok')
        plt.plot(theta, Model, label='x0=%.2f, sg=%.2f'%(out.params['x0'].value, out.params['sigma'].value))
        plt.legend()
        plt.savefig(image_filename)
        # plt.show()

    print("Sigma_al \t sigma_ni")
    print("Sigma_al \t sigma_ni", file=orig_stdout)

    stop = time.time()
    print('Total time consumed is %.1f sec\n\n' % (stop - start))
    print('Total time consumed is %.1f sec\n\n' % (stop - start), file=orig_stdout)

    sys.stdout = orig_stdout
    fout.close()

    return 0


def twogauss_fit(prm_name):
    orig_stdout = sys.stdout
    fout_name = 'tests/'+'pyXSW_' + str(int(time.time()))
    fout = open(fout_name + '.log', mode='a')    # log-file, unique for each trial
    sys.stdout = fout

    print('Output will be written to:\n', file=orig_stdout)
    print(fout_name, file=orig_stdout)

    initials = prm_name    # reads all fitting parameters from .prm-file
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
    bragg = float(num_params['bragg'])

    # print (params['normalize'])

    # fitting parameters
    normalize = str_params['normalize']
    if normalize == 'true':
        normalize = True
    else:
        normalize = False

    scan_zmin_zmax = str_params['scan_zmin_zmax']
    if scan_zmin_zmax.lower() == 'true':
        scan_zmin_zmax = True
    else:
        scan_zmin_zmax = False

    try:
        window = num_params['window']
    except KeyError:
        window = 1

    # fitting initial conditions
    angle_slope = float(num_params['angle_slope'])

    amplitude1 = float(num_params['amplitude1'])
    sigma1 = float(num_params['sigma1'])
    x01 = float(num_params['x01'])

    amplitude2 = float(num_params['amplitude2'])
    sigma2 = float(num_params['sigma2'])
    x02 = float(num_params['x02'])

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

    # TODO: organise errors export
    # theta, real_data = get_dat(data, normalize=normalize, bragg=bragg, template=template)
    # errors = np.ones(len(theta))
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
    out = minimize(gaussian_residual_two, num_params, args=(table_opt, theta, real_data, errors))

    Model = intensity_two(table_opt, theta,
                          out.params['amplitude1'],
                          out.params['amplitude2'],
                          out.params['sigma1'],
                          out.params['sigma2'],
                          out.params['x01'],
                          out.params['x02'],
                          out.params['angle_slope'],
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

    chisquared = 1.0 / (len(theta) - 6) * sum((Model - real_data)**2 / errors**2)
    print("\nChi-squared=%f" % chisquared)
    print("\nChi-squared=%f" % chisquared, file=orig_stdout)
    print("\nR-factor=%f"%rfactor)
    print("\nR-factor=%f"%rfactor, file=orig_stdout)

    image_filename = fout_name + '.png'
    print('\nImage is saved into: %s' % image_filename)
    print('\nImage is saved into: %s' % image_filename, file=orig_stdout)

    plt.plot(theta, real_data, 'ok')
    plt.plot(theta, Model, label='x01=%.2f, sg1=%.2f, x02=%.2f, sg2=%.2f' %
                                 (out.params['x01'].value, out.params['sigma1'].value,
                                  out.params['x02'].value, out.params['sigma2'].value))
    plt.legend()
    plt.savefig(image_filename)
    # plt.show()

    stop = time.time()
    print('Total time consumed is %.1f sec\n\n' % (stop - start))
    print('Total time consumed is %.1f sec\n\n' % (stop - start), file=orig_stdout)

    sys.stdout = orig_stdout
    fout.close()

    return 0