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

import seaborn as sea
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats 

plt.ion() # this makes matplotlib SHOW the plots as soon as they are created

parser = argparse.ArgumentParser()
args = parser.parse_args()

### Set LOCATION and analysis before running script to match needs ###
LOCATION = args.loc



################################
######## import data   #########
################################


PATH = os.path.expanduser('~')

data_dir = PATH+'/Dropbox (LewPeaLab)/BEHAVIOR/monitain'

fnames = glob.glob(data_dir+'/v*/*.csv')

# remove test subjects
fnames = [ fn for fn in fnames if 's999' not in fn ]

df_list = []
for i, fn in enumerate(fnames): 
    subj_df = pd.read_csv(fn)
    # add subj and trial num to df
    subj_df['subj'] = 's{:02d}'.format(i+1)
    subj_df['trial'] = range(subj_df.shape[0])
    df_list.append(subj_df)

df = pd.concat(df_list,ignore_index=True)

