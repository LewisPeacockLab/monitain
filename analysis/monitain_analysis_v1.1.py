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
##### Import data & setup ######
################################


PATH = os.path.expanduser('~')
data_dir = PATH+'/Dropbox (LewPeaLab)/BEHAVIOR/monitain'
FIGURE_PATH = PATH + '/monitain/analysis/output/'

fnames = glob.glob(data_dir+'/v*/*.csv')

# remove test subjects
fnames = [ fn for fn in fnames if 's999' not in fn ]

df_list = []
for i, fn in enumerate(fnames): 
    subj_df = pd.read_csv(fn, index_col=0)
    # add subj and trial num to df
    subj_df['subj'] = 's{:02d}'.format(i+1)
    subj_df['trial'] = range(subj_df.shape[0])
    df_list.append(subj_df)

df_main = pd.concat(df_list,ignore_index=False)
df_main = df_main.reset_index()
# Make a copy to replace inaccurate responses with no RT
df_main_copy = df_main
# Use for backup

#Replace rt with nan if probe resp was incorrect 
for i in df_main.index: 
	for probe in range(0,15): 
		if df_main.loc[i, 'probe{:d}_acc'.format(probe)] == 0: 
			df_main.at[i, 'rtProbe{:d}'.format(probe)] = np.nan


#Master list of all rt probe titles
rtProbes = []
for rt in range(0,14): #get rid of probe14 bc you'll never look at it 
	rtProbes.append('rtProbe{:d}'.format(rt))

accCols = []
for acc in range(0,15): #get rid of probe14 bc you'll never look at it 
	accCols.append('probe{:d}_acc'.format(acc))


#####################################
#######  By block accuracy  #########
#####################################


###BASELINE

# Find mean rt for baseline blocks (CORRECT RESPONSES)
block1_rt = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]\
	.groupby(['subj', 'block'])['rtProbe0'].mean()
block8_rt = df_main[df_main['block'] == 8].groupby(['subj', 'block'])['rtProbe0'].mean()

#dfs for violin plots
block1 = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]
block8 = df_main[(df_main['block'] == 8) & (df_main['probe0_acc']== 1)]
baseline_df = pd.concat([block1, block8], axis = 0).reset_index()#axis = 0 for horiz cat, = 1 for vert cat

# Compare baseline 1 to baseline 8 
ax = sea.violinplot(x='subj', y = 'rtProbe0', hue = 'block', data=baseline_df, palette = "Greens", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'baseline_compare.png', dpi = 600)
plt.close()



###MAINTAIN

block2 = df_main[(df_main['block'] == 2)]
block3 = df_main[(df_main['block'] == 3)]


block2_df = pd.concat([ block2['subj'], block2[rtProbes].mean(axis=1)], axis=1)
block2_df['block'] = 2
block3_df = pd.concat([ block3['subj'], block3[rtProbes].mean(axis=1)], axis=1)
block3_df['block'] = 3

maintain_df = pd.concat([block2_df, block3_df], axis=0)
maintain_df.columns = ['subj', 'meanTrial_rt', 'block']

ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=maintain_df, palette = "Blues", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'maintain_compare.png', dpi = 600)
plt.close()




#####################################
############  Figures  ##############
#####################################







