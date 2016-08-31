from __future__ import print_function
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from export import *
from optimization import *
from lmfit import Parameters, minimize
from procedures import onegauss_fit, twogauss_fit


prm_name = sys.argv[1]

onegauss_fit(prm_name)
# twogauss_fit(prm_name)
