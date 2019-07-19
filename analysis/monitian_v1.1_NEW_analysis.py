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

df_main = pd.concat(df_list,ignore_index=False)
df_main = df_main.reset_index()
# Make a copy to replace inaccurate responses with no RT
df_main_copy = df_main
# Use for backup

#Record PM accuracy per trial
for trial in df_main.index: 
	probeNum = df_main.loc[trial, 'n_probes']
	if df_main.loc[trial, 'index'] == 0: #something wrong with first trial 
		pass
	elif df_main.loc[trial, 'probe{:d}_acc'.format(probeNum-1)] == 1: #correct PM resp
		df_main.loc[trial, 'acc'] = 1
	else:
		print
		df_main.loc[trial, 'acc'] = 0

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


######### BY BLOCK TYPE DATAFRAMES #########

def createBlockDFs(str1, str2):
	block_name = df_main[(df_main['block'] == str1) | (df_main['block'] == str2)]
	block_name_df = pd.concat([ block_name['subj'], block_name['block'], block_name['acc'], block_name[rtProbes].mean(axis=1), block_name[accCols].mean(axis=1)], axis=1)
	block_name_df.columns = ['subj', 'block', 'pm_acc', 'meanTrial_rt', 'og_acc']
	return block_name_df

block_maintain_df = createBlockDFs("('maintain1', 2)", "('maintain2', 5)")
block_monitor_df = createBlockDFs("('monitor1', 3)", "('monitor2', 6)")
block_mnm_df = createBlockDFs("('mnm1', 4)", "('mnm2', 7)")


######### BY SUBJECT FIGURES #########


## PM Accuracy
def bySubj_pmAcc(block_name_df, blockStr, colorPalette):

	ax = sea.barplot(x='subj', y= 'pm_acc', data=block_name_df, palette="Blues", ci = None)
	plt.xlabel('Subject')
	plt.ylabel('PM accuracy')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'bysubj_' + blockStr + '_pmacc.png', dpi = 600)
	plt.close()

bySubj_pmAcc(block_maintain_df, 'maintain')
bySubj_pmAcc(block_monitor_df, 'monitor')
bySubj_pmAcc(block_mnm_df, 'mnm')


## OG Accuracy
def bySubj_ogAcc(block_name_df, blockStr, colorPalette): 
	ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=block_name_df, palette="Reds", ci = None)
	plt.xlabel('Subject')
	plt.ylabel('OG accuracy')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'bySubj' + blockStr + '_ogacc.png', dpi = 600)
	plt.close()

bySubj_ogAcc(block_monitor_df, 'maintain', "Blues")
bySubj_ogAcc(block_monitor_df, 'monitor', "Reds")
bySubj_ogAcc(block_monitor_df, 'mnm', "Purples")

## PM cost
def bySubj_pmCost(block_name_df, blockStr): 

# Reaction times
def bySubj_rt(block_name_df, blockStr): 