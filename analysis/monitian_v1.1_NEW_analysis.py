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

def createBlockDFs(str1, str2, blockType):
	block_name = df_main[(df_main['block'] == str1) | (df_main['block'] == str2)]	
	block_name_df = pd.concat([ block_name['subj'], block_name['block'], block_name['acc'], block_name[rtProbes].mean(axis=1), block_name[accCols].mean(axis=1)], axis=1)
	block_name_df['blockType'] = blockType
	block_name_df.columns = ['subj', 'block', 'pm_acc', 'meanTrial_rt', 'og_acc', 'blockType',]	
	return block_name_df

block_baseline_df = createBlockDFs("('base1', 1)", "('base2', 8)", "Baseline")
block_maintain_df = createBlockDFs("('maintain1', 2)", "('maintain2', 5)", "Maintain")
block_monitor_df = createBlockDFs("('monitor1', 3)", "('monitor2', 6)", "Monitor")
block_mnm_df = createBlockDFs("('mnm1', 4)", "('mnm2', 7)", "MnM")



######### CALCULATE PM COST #########

# Calculate base cost by subj
base_cost = block_baseline_df.groupby('subj')['meanTrial_rt'].mean()

def pmCost(block_name_df, blockStr): 
	block_cost = block_name_df.groupby('subj')['meanTrial_rt'].mean()
	block_cost_PM = (block_cost - base_cost).to_frame().reset_index()
	block_cost_PM['blockType'] = blockStr
	return block_cost_PM

maintain_cost_PM = pmCost(block_maintain_df, 'Maintain')
monitor_cost_PM = pmCost(block_monitor_df, 'Monitor')
mnm_cost_PM = pmCost(block_mnm_df, 'MnM')

pmCost_df = pd.concat([maintain_cost_PM, monitor_cost_PM, mnm_cost_PM], axis = 0)
pmCost_df.columns = (['subj', 'pm_cost', 'blockType']) #PM cost is essentially the RT cost

######### BY SUBJECT FIGURES #########

## PM Accuracy
def bySubj_pmAcc(block_name_df, blockStr, colorPalette):

	ax = sea.barplot(x='subj', y= 'pm_acc', data=block_name_df, palette=colorPalette, ci = None)
	plt.xlabel('Subject')
	plt.ylabel('PM accuracy')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'bySubj_' + blockStr + '_pmacc.png', dpi = 600)
	plt.close()

bySubj_pmAcc(block_maintain_df, 'maintain', "Blues")
bySubj_pmAcc(block_monitor_df, 'monitor', "Reds")
bySubj_pmAcc(block_mnm_df, 'mnm', "Purples")


## OG Accuracy
def bySubj_ogAcc(block_name_df, blockStr, colorPalette): 
	ax = sea.barplot(x='subj', y= 'og_acc', data=block_name_df, palette=colorPalette, ci = None)
	plt.xlabel('Subject')
	plt.ylabel('OG accuracy')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'bySubj_' + blockStr + '_ogacc.png', dpi = 600)
	plt.close()

bySubj_ogAcc(block_baseline_df, 'baseline', "Greens")
bySubj_ogAcc(block_monitor_df, 'maintain', "Blues")
bySubj_ogAcc(block_monitor_df, 'monitor', "Reds")
bySubj_ogAcc(block_monitor_df, 'mnm', "Purples")


## Reaction time
def bySubj_rt(block_name_df, blockStr, colorPalette):
	ax = sea.violinplot(x='subj', y = 'meanTrial_rt', data=block_name_df, palette = colorPalette, cut = 0)
	# Cut = 0 so range is limited to observed data
	plt.xlabel('Subject')
	plt.ylabel('Reaction time (s)')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'bySubj_' + blockStr + '_rt.png', dpi = 600)
	plt.close()

bySubj_rt(block_baseline_df, 'baseline', "Greens")
bySubj_rt(block_monitor_df, 'maintain', "Blues")
bySubj_rt(block_monitor_df, 'monitor', "Reds")
bySubj_rt(block_monitor_df, 'mnm', "Purples")



######### ALL BLOCKS DATAFRAME #########

all_df = pd.concat([block_baseline_df, block_maintain_df, block_monitor_df, block_mnm_df], axis = 0)
all_df_minusBase = pd.concat([block_maintain_df, block_monitor_df, block_mnm_df], axis = 0)
# If just looking at all_df_minusBase, need to redo exclusion criteria for that



######### EXCLUDING DATA #########

#Replace RT with nan in resp was quicker than 0.05 secs, also replace pm acc with nan
#Cut off based on findings from "Eye Movements and Visual Encoding during Scene Perception"
#Rayner et al., 2009
for trial in all_df.index: 
	if all_df.loc[trial, 'meanTrial_rt'] < 0.05: 
		## print all_df_minusBase.loc[trial, 'meanTrial_rt']
		all_df.at[trial, 'meanTrial_rt'] = np.nan
		all_df.at[trial, 'pm_acc'] = np.nan



######### ALL BLOCKS FIGURES #########

## Custom color palette
my_pal = ['#aedea7', #Green
		'#abd0e6', #Blue 
		##'#3787c0', #Blue 
		'#fca082', #Red 
		##'#e32f27', #Red
		'#c6c7e1', #Purple
		##'#796eb2', #Purple
		##'#37a055'  #Green
		  ]

#### FUNCTIONS to create figures

## PM accuracy
def allSubj_pmAcc():
	ax = sea.barplot(x='blockType', y= 'pm_acc', data=all_df, palette=my_pal)
	plt.xlabel('Block Type')
	plt.ylabel('PM accuracy')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'allSubj_pmAcc.png', dpi = 600)
	plt.close()

## OG accuracy
def allSubj_ogAcc():
	ax = sea.barplot(x='blockType', y= 'og_acc', data=all_df, palette=my_pal)
	plt.xlabel('Block Type')
	plt.ylabel('OG accuracy')
	sea.despine()
	plt.savefig(FIGURE_PATH + 'allSubj_ogAcc.png', dpi = 600)
	plt.close() 

## Reaction times
def allSubj_rt(): 
	ax = sea.violinplot(x='blockType', y = 'meanTrial_rt', data=all_df, palette = my_pal, cut = 0)
	plt.xlabel('Block Type')
	plt.ylabel('Reaction time (s)')
	ax.tick_params(axis='x', labelsize=7)
	sea.despine()
	plt.savefig(FIGURE_PATH + 'allSubj_rt.png', dpi = 600)
	plt.close()

## PM cost
def allSubj_pmCost():
	ax = sea.barplot(x='blockType', y= 'pm_cost', data=pmCost_df)
	plt.xlabel('Block')
	plt.ylabel('PM cost (s)')
	ea.despine()
	plt.savefig(FIGURE_PATH + 'allSubj_PMCOST.png', dpi = 600)
	plt.close()


#### CREATE figures for all subjects
allSubj_pmAcc()
allSubj_ogAcc()
allSubj_rt()
allSubj_pmCost()