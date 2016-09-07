from __future__ import print_function

import sys

from functions import *


def onegaussian_fit(num_param, str_param, project_name='onegauss'):
    try:
        project_name = str_param['project_name']
    except KeyError:
        pass
    # will look like project_name_2016-8-31-12-58-48
    prefix = project_name + '_' + propertime()
    os.mkdir(prefix)
    os.chdir(prefix)

    log = open('%s.log' % prefix, mode='w')
    screen = sys.stdout

    graph_print = boolean_translate(str_param, 'graph_print')
    table_print = boolean_translate(str_param, 'table_print')
    dat_print = boolean_translate(str_param, 'dat_print')
    normalize = boolean_translate(str_param, 'normalize')

    table = str_param['table']
    xserver = str_param['xserver']
    table = get_grd(table, xserver=xserver)

    bragg = num_param['bragg'].value
    template = str_param['template']
    real_data = str_param['data']
    if template == 'xy':
        theta, yelid = get_dat(real_data, bragg=bragg, normalize=normalize, template=template)
        yelid_errors = None
    elif template == 'xyy' or template == 'xyyerror':
        theta, yelid, yelid_errors = get_dat(real_data, bragg=bragg, normalize=normalize, template=template)
    elif template == 'xxerroryyerror':
        theta, theta_error, yelid, yelid_errors = get_dat(real_data, bragg=bragg, normalize=normalize,
                                                          template=template)
    else:
        raise ValueError('.dat file was not read')

    logprint(log, screen, """
    Data succesfully read from:
    prefix = %s
    table = %s
    data  = %s

    bragg = %f
    normalize = %s
    graph_print = %s
    table_print = %s
    """ % (prefix, str_param['table'], str_param['data'], bragg, normalize, graph_print, table_print))

    # now we have yelid with correct angles (not relative to bragg),
    # errors and standing wave table as np.array() with correct order

    logprint(log, screen, "\n")
    logprint(log, screen, """
************************
* MINIMIZATION STARTED *
************************""")

    logprint(log, screen, """
Initial parameters are:\n""")

    sys.stdout = log
    num_param.pretty_print()
    print('\n')
    for key in str_param.keys():
        print(key, str_param[key], sep=':\t')
    sys.stdout = screen
    num_param.pretty_print()
    print('\n')
    for key in str_param.keys():
        print(key, str_param[key], sep=':\t')

    print("Chisquared\trfactor\tx0\tsigma\tamplitude")

    start = time.time()
    out = minimize(residual_onegaussian, num_param, args=(table, theta, yelid, yelid_errors))
    stop = time.time()

    logprint(log, screen, """
MINIMIZATION FINISHED

Time consumed is: %.2f s

""" % (stop - start))

    logprint(log, screen, """
Fitted parameters are:
""")

    sys.stdout = log
    out.params.pretty_print()
    sys.stdout = screen
    out.params.pretty_print()

    model, ibar = intensity_onegaussian(table,
                                        theta,
                                        out.params['amp'],
                                        out.params['sigma'],
                                        out.params['x0'],
                                        out.params['zmax'],
                                        out.params['angle_slope'],
                                        out.params['zmin'],
                                        get_ibar=True
                                        )

    if yelid_errors is not None:  # if we have errors
        chisquared = np.sum((yelid - model) ** 2 / yelid_errors ** 2) / \
                     (len(theta) - 4)  # that must be chi-squared criteria with errors
    else:
        chisquared = np.sum((yelid - model) ** 2) / \
                     (len(theta) - 4)  # that must be chi-squared criteria with similar errors

    rfactor = sum(abs(model - yelid) / sum(yelid))

    logprint(log, screen, """

Rfactor:\t\t %f
Chisquared:\t\t %f

    """ % (rfactor, chisquared))

    if dat_print:
        fout = open('data_%s.dat' % prefix, 'w')
        for i in range(len(theta)):
            print(theta[i], model[i], file=fout, sep='\t')
        fout.close()

    if graph_print:
        plt.plot(theta, model, 'b')
        plt.plot(theta, yelid, 'ko')
        plt.title('x0=%.1f, sigma=%.1f' % (out.params['x0'], out.params['sigma']))
        plt.savefig('yelid_%s.png' % prefix)
        plt.clf()

    if table_print:
        zmin = out.params['zmin']
        zmax = out.params['zmax']
        sigma = out.params['sigma']
        x0 = out.params['x0']
        amp = out.params['amp']
        angle_slope = out.params['angle_slope']
        gauss = lambda coord: (amp + angle_slope * 0) * np.exp(-(coord - x0) ** 2 / 2.0 / sigma ** 2)

        plt.subplot(3, 1, 1)

        plt.plot(theta, model, 'b')
        plt.plot(theta, yelid, 'ko')
        plt.title('x0=%.1f, sigma=%.1f' % (out.params['x0'], out.params['sigma']))

        plt.subplot(3, 1, 2)
        plt.imshow(ibar,
                   aspect='auto',
                   extent=(zmin, zmax, np.min(theta), np.max(theta)))
        # plt.title('x0=%.1f, sigma=%.1f'%(out.params['x0'], out.params['sigma']))
        plt.savefig('tables_%s.png' % prefix)
        plt.colorbar()

        plt.subplot(3, 1, 3)
        x = np.linspace(zmin, zmax, int(abs(zmax - zmin) * 2))
        y = gauss(x)
        plt.plot(x, y)

    print(os.getcwd())
    log.close()
    return out.params, theta, model, yelid, chisquared, rfactor


parameters = sys.argv[1]

# os.chdir('/home/errorochka/Dropbox/DESY/pyXSW/tests/one_gaussian')
num_param, str_param = get_initials(parameters)
params, theta, model, yelid, chisquared, rfactor = onegaussian_fit(num_param, str_param)
plt.show()
