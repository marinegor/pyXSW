from __future__ import print_function
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
from optimization import *
from export import *
import time
import sys
N_al = 11
N_ni = 11
fin = open('/home/errorochka/Dropbox/DESY/scripts/grid_gen/getTER_sw_query_post.pl').read().split('\n')
os.chdir('/home/errorochka/Dropbox/DESY/scripts/grid_gen/')

log = open('logfile.txt', mode='w')

top = fin[94]
al = fin[97]
ni = fin[98]

sigma_al = [str(i) for i in np.linspace(0, 5, N_al)]
sigma_ni = [str(round(i, 2)) for i in np.linspace(0, 5, N_ni)]

# sigma_al = ['7.2']
# sigma_ni = ['15']

for sal in sigma_al:
    for sni in sigma_ni:
        filename = 'al' + sal + '_ni_' + sni + '.pl'
        fout = open(filename, mode='w')

        print('Running for: %s'%os.getcwd(), file=log)
        print('Al_sigma = %s \t Ni_sigma = %s \n' % (sal, sni), file=log)

        al = fin[97]
        ni = fin[98]

        al = al.split('=')
        al[-1] = str(sal)
        al = '='.join(al)

        ni = ni.split('=')
        ni[-1] = str(sni)
        ni = '='.join(ni)

        for i in range(len(fin)):
            if i == 97:
                print(al, file=fout)
            elif i == 98:
                print(ni, file=fout)

            else:
                print(fin[i], file=fout)

        fout.close()
        os.system('chmod +x ' + filename)
        os.system('./' + filename)

        grid_name = [fle for fle in os.listdir(os.getcwd()) if fle[-4:] == '.grd'][0]
        grid = get_grd(grid_name)

        prm_in = open('initial.prm').read().split('\n')
        prm_out = open('initial.prm', mode='w')
        prm_in[0] = 'table=' + grid_name
        print(*prm_in, sep='\n', file=prm_out)
        prm_out.close()

        os.system('python '
                  '/home/errorochka/Dropbox/DESY/pyXSW/main.py '
                  '/home/errorochka/Dropbox/DESY/scripts/grid_gen/initial.prm ' + str(sal) + ' ' + str(sni))

        os.system('rm ' + grid_name)

log.close()
