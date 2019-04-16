##################################
##################################
####### monitain ANALYSES ########
######## Katie Hedgpeth ##########
########## April 2019 ############
##################################
##################################


import os 
import glob
import argparse
import numpy as np
import pandas as pd
import math

import seaborn as sea
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats 

plt.ion() # this makes matplotlib SHOW the plots as soon as they are created

parser = argparse.ArgumentParser()
parser.add_argument('--loc',default='lab',type=str,choices=['lab','home'])
args = parser.parse_args()

### Set LOCATION and analysis before running script to match needs ###
#LOCATION = args.loc



################################
######## import data   #########
################################


PATH = os.path.expanduser('~')

if LOCATION == 'home':
    data_dir = PATH+'/repo/DATA'
elif LOCATION == 'lab':
    data_dir = PATH+'/Dropbox (LewPeaLab)/BEHAVIOR/RepO/data_output'


fnames = glob.glob(data_dir+'/vers*/csvs/*.csv')

# remove test subjects
fnames = [ fn for fn in fnames if 's999' not in fn ]

