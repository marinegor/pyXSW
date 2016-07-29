from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from optimization import *
from export import *

initials = '/home/errorochka/Dropbox/DESY/pyXSW/tests/optimization/initial.prm'
# theta = '/home/errorochka/Dropbox/DESY/pyXSW/tests/optimization/test_ourlattice.dat'
# table = '/home/errorochka/Dropbox/DESY/pyXSW/tests/optimization/test_ourlattice.grd'

theta = '/home/errorochka/Dropbox/DESY/pyXSW/tests/optimization/SGS_highhum_K.dat'
table = '/home/errorochka/Dropbox/DESY/pyXSW/tests/optimization/SGS_highhum_K.grd'

params = get_initials(initials)
theta, real_data = get_dat(theta)
table_opt = get_grd(table)
# pass
out = minimize(gaussian_residual, params, args=(table_opt, theta, real_data))

Model = intensity(table_opt, theta,
                  out.params['amplitude'],
                  out.params['sigma'],
                  out.params['x0'],
                  out.params['angle_slope'],
                  out.params['bragg'],
                  out.params['zmin'],
                  out.params['zmax'])

rfactor = sum(abs(Model-real_data))/sum(real_data)

out.params.pretty_print()
print("R-factor=%f"%rfactor)

plt.plot(theta, real_data, 'ok')
plt.plot(theta, Model)
plt.show()

