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

df_main = pd.concat(df_list,ignore_index=True)

blockType_grouped = df_main.groupby(['subj', 'block'])


def get_block_type(row): 
	block_type = row['block']

#####################################
#######  By probe accuracy  #########
#####################################

# Find mean rt for baseline 1
block1_rt = df_main[df_main['block'] == 1].groupby('subj')['rtProbe0'].mean()
block8_rt = df_main[df_main['block'] == 8].groupby('subj')['rtProbe0'].mean()

#15 empty probe groups
probe = [0]*16  
probeGroup = [0]*16 

for i in range(1,16): 
	probe[i] = df_main[df_main['n_probes'] == i]
	probeGroup[i] = probe[i].groupby(['subj', 'block'])['rtProbe{:d}'.format(i-1)].mean() 
	#probe[1] is for 1 probe trials


#Try this? 
probe[1].groupby(['subj', 'block'])['rtProbe0'].mean()    

for i 


for i in range(15): 
	probe[i] = df_main[df_main['n_probes'] == i]





df_main.groupby(['subj','block'])['rtProbe0'].mean()
