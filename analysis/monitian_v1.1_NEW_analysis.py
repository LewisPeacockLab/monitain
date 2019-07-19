##################################
##################################
####### monitain ANALYSES ########
########### for v.1.1 ############
######## Katie Hedgpeth ##########
########### July 2019 ############
##################################
##################################

import os 
import glob
import argparse
import numpy as np
import pandas as pd

import seaborn as sea
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats 

plt.ion() # this makes matplotlib SHOW the plots as soon as they are created

################################
##### Import data & setup ######
################################


PATH = os.path.expanduser('~')
data_dir = PATH+'/Dropbox (LewPeaLab)/BEHAVIOR/monitain/'
FIGURE_PATH = PATH + '/monitain/analysis/output/'

fnames = glob.glob(data_dir+'/v1.1/*.csv')

# remove test subjects
fnames = [ fn for fn in fnames if 's999' not in fn ]

df_list = []
for i, fn in enumerate(fnames): 
    subj_df = pd.read_csv(fn, index_col=0)
    # add subj and trial num to df
    subj_df['subj'] = 's{:02d}'.format(i+1)
    subj_df['trial'] = range(subj_df.shape[0])

    if 'v1.0' in fn: 
    	vers = 'vers1.0'
    elif 'v1.1' in fn: 
    	vers = 'vers1.1'

    subj_df['version'] = vers
    df_list.append(subj_df)