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

#blockType_grouped = df_main.groupby(['subj', 'block'])


#def get_block_type(row): 
#	block_type = row['block']

#####################################
#######  By block accuracy  #########
#####################################


###BASELINE
# Find mean rt for baseline blocks (ALL RESPONSES)
#block1_rt = df_main[df_main['block'] == 1].groupby(['subj', 'block'])['rtProbe0'].mean()
#block8_rt = df_main[df_main['block'] == 8].groupby(['subj', 'block'])['rtProbe0'].mean()

# Find mean rt for baseline blocks (CORRECT RESPONSES)
block1_rt = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]\
	.groupby(['subj', 'block'])['rtProbe0'].mean()
block8_rt = df_main[df_main['block'] == 8].groupby(['subj', 'block'])['rtProbe0'].mean()

#dfs for violin plots
block1 = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]
block8 = df_main[(df_main['block'] == 8) & (df_main['probe0_acc']== 1)]
baseline_df = pd.concat([block1, block8], axis = 0).reset_index()#axis = 0 for horiz cat, = 1 for vert cat

df_main[ (df_main['probe{:d}_acc'.format(probe)])]
df_main['probe{:d}_acc'.format(probe)]

df_main_copy.loc[1,'probe{:d}_acc'.format(probe)]

for i in df_main_copy.index: 
	for probe in range(0,15): 
		if df_main_copy.loc[i, 'probe{:d}_acc'.format(probe)] == 0: 
			df_main_copy.at[i, 'rtProbe{:d}'.format(probe)] = np.nan

# Create a column for trial accuracy
#df_main['trialAcc'] = 0

#master list of all rt probe titles
rtProbes = []
for rt in range(0,14): #get rid of probe14 bc you'll never look at it 
	rtProbes.append('rtProbe{:d}'.format(rt))

accCols = []
for acc in range(0,15): #get rid of probe14 bc you'll never look at it 
	accCols.append('probe{:d}_acc'.format(acc))

### MAINTAIN
##### KEEP THIS!!!
block2 = df_main_copy[(df_main_copy['block'] == 2) ]
#& (df_main['probe0_acc']== 1)]
block3 = df_main_copy[(df_main_copy['block'] == 3) ]
#& (df_main['probe0_acc']== 1)]


#block2[rtProbes].mean(axis=1)
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
#Need to take out incorrect responses!!

########KEEP THIS!







#15 empty probe groups
n=16
probe = [0]*n 
probeGroup = [[[] for i in range(n)] for i in range(n)]

#ogProbes = [[[] for i in range(n)] for i in range(n)]
ogProbes = pd.DataFrame()

for i in range(1,16): 
	probe[i] = df_main[df_main['n_probes'] == i]
	for j in range(i): 
		#probeGroup[i][j] = probe[i].groupby(['subj', 'block'])['rtProbe{:d}'.format(j)].mean()
		probeGroup[i][j] = probe[i].groupby(['subj', 'block'])['rtProbe{:d}'.format(j)] 
		if i in range(8,16):
			if j+1 < i: #if j is not the last probe
				#ogProbes[i].append(probeGroup[i][j])
				ogProbes.append(probeGroup[i][j])
				#ogProbes 8 to 15 exists


#####THIS WORKS
big_og_df_maintain = pd.DataFrame()
for i in range(8,16): 
	og_rts_maintain = df_main[(df_main['n_probes'] == i) & ((df_main['block'] == 2) | (df_main['block'] == 3))].groupby(['subj', 'block'])[rtProbes[:(i-1)]].mean()
	#i-1 so you don't get final probe with PM target
	big_og_df_maintain = big_og_df_maintain.append(og_rts_maintain)
	#big_og_df_maintain = big_og_df_maintain.reset_index()

	og_df_maintain = big_og_df_maintain.groupby(['subj', 'block'])[rtProbes].mean()
	og_df_maintain = og_df_maintain.reset_index()

	ax = sea.violinplot(x='subj', y= og_df_maintain[rtProbes], hue='block', data = og_df_maintain, 
		palette = "Blues", cut = 0)

	ax = sea.violinplot(x='subj', y = 'rtProbe0', hue = 'block', data=baseline_df, 
		palette = "Greens", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'baseline_compare.png', dpi = 600)
plt.close()
	#big df of all ogs
#####THIS WORKS

	print rtProbes[:i]


df_main.groupby(['n_probes','subj'])[rtProbes].mean()

	#probe[1] is for 1 probe trials
	# i is number of probes, j is rt for probe # j is series of 1,2,3,4...n_probes for rtProbe0 to rtProbe14
	#lowest is probeGroup[1][0]
	#highest is probeGroup[15][14]

#n_probes = 1 to 7 are catch trials
# so probeGroup[1][x] to probeGroup[7][x] don't matter


#Try this? 
probe[1].groupby(['subj', 'block'])['rtProbe0'].mean()    



for i in range(15): 
	probe[i] = df_main[df_main['n_probes'] == i]





df_main.groupby(['subj','block'])['rtProbe0'].mean()






#####################################
############  Figures  ##############
#####################################

# Compare baseline 1 to baseline 8 
ax = sea.violinplot(x='subj', y = 'rtProbe0', hue = 'block', data=baseline_df, palette = "Greens", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'baseline_compare.png', dpi = 600)
plt.close()

