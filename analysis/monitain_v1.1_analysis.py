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

plt.ion() # This makes matplotlib SHOW the plots as soon as they are created

################################
##### Import data & setup ######
################################

PATH = os.path.expanduser('~')
data_dir = PATH+'/Dropbox (LewPeaLab)/BEHAVIOR/monitain'
FIGURE_PATH = PATH + '/monitain/analysis/output/'

fnames = glob.glob(data_dir+'/v1.1/*.csv')
fnames = sorted(fnames)

# Remove test subjects
fnames = [ fn for fn in fnames if 's999' not in fn ]

df_list = []
for i, fn in enumerate(fnames): 
	subj_df = pd.read_csv(fn, index_col=0)
	# Add subj and trial num to df
	subj_df['subj'] = 's{:02d}'.format(i+1)
	subj_df['trial'] = range(subj_df.shape[0])

	if 'v1.0' in fn: 
		vers = 'vers1.0'
	elif 'v1.1' in fn: 
		vers = 'vers1.1'

	subj_df['version'] = vers
	subj_df = subj_df[1:] #get rid of first line because something is off
	df_list.append(subj_df)

##df_main = pd.concat(df_list,ignore_index=False, sort=False)
df_main = pd.concat(df_list,ignore_index=False)
df_main = df_main.reset_index()
# Make a copy to replace inaccurate responses with no RT
df_main_copy = df_main
# Use for backup

# Record PM accuracy per trial
for trial in df_main.index: 
	probeNum = df_main.loc[trial, 'n_probes']
	if df_main.loc[trial, 'index'] == 0: #something wrong with first trial 
		pass
	elif df_main.loc[trial, 'probe{:d}_acc'.format(probeNum-1)] == 1: #correct PM resp
		df_main.loc[trial, 'acc'] = 1
	else:
		print
		df_main.loc[trial, 'acc'] = 0

# Replace rt with nan if probe resp was incorrect 
for i in df_main.index: 
	for probe in range(0,15): 
		if df_main.loc[i, 'probe{:d}_acc'.format(probe)] == 0: 
			df_main.at[i, 'rtProbe{:d}'.format(probe)] = np.nan

# Master list of all rt probe titles
rtProbes = []
for rt in range(0,14): #get rid of probe14 bc you'll never look at it 
	rtProbes.append('rtProbe{:d}'.format(rt))

accCols = []
for acc in range(0,15): #get rid of probe14 bc you'll never look at it 
	accCols.append('probe{:d}_acc'.format(acc))



######### BY BLOCK TYPE DATAFRAMES #########

def createBlockDFs(str1, str2, blockType):
	block_name = df_main[(df_main['block'] == str1) | (df_main['block'] == str2)]	
	##block_name_df = pd.concat([ block_name['subj'], block_name['block'], block_name['acc'], block_name[rtProbes].mean(axis=1), block_name[accCols].mean(axis=1)], axis=1 , sort=False)
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

##pmCost_df = pd.concat([maintain_cost_PM, monitor_cost_PM, mnm_cost_PM], axis = 0, sort=False)
pmCost_df = pd.concat([maintain_cost_PM, monitor_cost_PM, mnm_cost_PM], axis = 0)
pmCost_df.columns = (['subj', 'pm_cost', 'blockType']) #PM cost is essentially the RT cost

def byTrial_pmCost(block_name_df): 
	pmCost = []
	for index, row in block_name_df.iterrows():  
		for subj, rt in base_cost.iteritems():  
			if row.subj == subj: 
				pmCost.append(row.meanTrial_rt - rt)
	block_name_df['pm_cost'] = pmCost

## Calculuate by trial PM cost
byTrial_pmCost(block_maintain_df)
byTrial_pmCost(block_monitor_df)
byTrial_pmCost(block_mnm_df)

####PICK UP HERE

######### BY SUBJECT FIGURES #########

## PM Accuracy
def bySubj_pmAcc(block_name_df, blockStr, colorPalette):

	ax = sea.barplot(x='subj', y= 'pm_acc', data=block_name_df, palette=colorPalette, ci = None)
	plt.xlabel('Subject')
	plt.ylabel('Accuracy')
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

##all_df = pd.concat([block_baseline_df, block_maintain_df, block_monitor_df, block_mnm_df], axis = 0, sort=False)
##all_df_minusBase = pd.concat([block_maintain_df, block_monitor_df, block_mnm_df], axis = 0, sort=False)
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

#### Export dataframes to CSVs
CSV_PATH = FIGURE_PATH + 'csvs'


######### EXPORT CSVs #########
## All DFs
all_df_averaged = all_df.groupby(['subj', 'blockType']).mean().reset_index(drop = False)
pmCost_df_averaged = pmCost_df.groupby(['subj', 'blockType']).mean().reset_index(drop = False)

fname_all = os.path.join(CSV_PATH, 'ALL.csv')
all_df_averaged.to_csv(fname_all, index = False)
fname_pm = os.path.join(CSV_PATH, 'PM_COST.csv')
pmCost_df_averaged.to_csv(fname_pm, index = False)

fname_all_byTrial = os.path.join(CSV_PATH, 'ALL_BYTRIAL.csv')
all_df.to_csv(fname_all_byTrial, index = False)



#fname_all_df = os.path.join(CSV_PATH, 'all_bySubj.csv')
#all_df.to_csv(fname_all_df, index = False)

### Baseline DF
#baseline_df = all_df[all_df['blockType'] == 'Baseline']
#fname_baseline_df = os.path.join(CSV_PATH, 'baseline.csv')
#baseline_df.to_csv(fname_baseline_df, index = False)

### Maintain DF
#maintain_df = all_df[all_df['blockType'] == 'Maintain']
#fname_maintain_df = os.path.join(CSV_PATH, 'maintain.csv')
#maintain_df.to_csv(fname_maintain_df, index = False)

### Monitor DF
#monitor_df = all_df[all_df['blockType'] == 'Monitor']
#fname_monitor_df = os.path.join(CSV_PATH, 'monitor.csv')
#monitor_df.to_csv(fname_monitor_df, index = False)

### MNM DF
#mnm_df = all_df[all_df['blockType'] == 'MnM']
#fname_mnm_df = os.path.join(CSV_PATH, 'mnm.csv')
#mnm_df.to_csv(fname_mnm_df, index = False)


## PM cost DFs
#fname_pmCost_df = os.path.join(CSV_PATH, 'pm_cost_bySubj.csv')
#pmCost_df.to_csv(fname_pmCost_df, index = False)

### Baseline DF - PM cost
#baseline_df_pmCost = pmCost_df[pmCost_df['blockType'] == 'Baseline']
#fname_baseline_PM_df = os.path.join(CSV_PATH, 'baseline_PM.csv')
#baseline_df_pmCost.to_csv(fname_baseline_PM_df, index = False)

### Maintain DF - PM cost
#maintain_df_pmCost = pmCost_df[pmCost_df['blockType'] == 'Maintain']
#fname_maintain_PM_df = os.path.join(CSV_PATH, 'maintain_PM.csv')
#maintain_df_pmCost.to_csv(fname_maintain_PM_df, index = False)

### Monitor DF - PM cost
#monitor_df_pmCost = pmCost_df[pmCost_df['blockType'] == 'Monitor']
#fname_monitor_PM_df = os.path.join(CSV_PATH, 'monitor_PM.csv')
#monitor_df_pmCost.to_csv(fname_monitor_PM_df, index = False)

### MnM DF - PM cost
#mnm_df_pmCost = pmCost_df[pmCost_df['blockType'] == 'MnM']
#fname_mnm_PM_df = os.path.join(CSV_PATH, 'mnm_PM.csv')
#mnm_df_pmCost.to_csv(fname_mnm_PM_df, index = False)


#### FUNCTIONS to create figures
all_df = all_df[(all_df['subj'] != 's18')] 

## PM accuracy
def allSubj_pmAcc():
	all_df_minusBase = all_df[all_df.blockType!= 'Baseline']
	ax = sea.barplot(x='blockType', y= 'pm_acc', data=all_df_minusBase, palette = my_pal[1:])
	plt.xlabel('Block type')
	plt.ylabel('Accuracy')
	plt.ylim(0,1.0)
	sea.despine()
	#plt.title('Average PM accuracy by block type')
	plt.savefig(FIGURE_PATH + 'allSubj_pmAcc.eps', dpi = 600)
	plt.close()

## OG accuracy
def allSubj_ogAcc():
	ax = sea.barplot(x='blockType', y= 'og_acc', data=all_df, palette=my_pal)
	plt.xlabel('Block type')
	plt.ylabel('OG accuracy')
	plt.ylim(0,1.0)
	sea.despine()
	#plt.title('Average OG accuracy by block type')
	plt.savefig(FIGURE_PATH + 'allSubj_ogAcc.eps', dpi = 600)
	plt.close() 

## Reaction times
def allSubj_rt(): 
	ax = sea.violinplot(x='blockType', y = 'meanTrial_rt', data=all_df, palette = my_pal, cut = 0)
	plt.xlabel('Block type')
	plt.ylabel('Reaction time (s)')
	ax.tick_params(axis='x', labelsize=7)
	sea.despine()
	#plt.title('Average RTs (s) by block type')
	plt.savefig(FIGURE_PATH + 'allSubj_rt.eps', dpi = 600)
	plt.close()

## PM cost
def allSubj_pmCost():
	ax = sea.barplot(x='blockType', y= 'pm_cost', data=pmCost_df, palette = my_pal[1:])
	plt.xlabel('Block type')
	plt.ylabel('PM cost (s)')
	sea.despine()
	#plt.title('Average PM cost by block type')
	plt.savefig(FIGURE_PATH + 'allSubj_PMCOST.eps', dpi = 600)
	plt.close()


#### CREATE figures for all subjects
allSubj_pmAcc()
allSubj_ogAcc()
allSubj_rt()
allSubj_pmCost()

new_df = all_df.groupby(['subj','blockType']).mean().reset_index(drop=False)
combine_df = new_df[(new_df['blockType'] == 'Monitor') |(new_df['blockType']=='Maintain')]
combine_df = combine_df.groupby('subj').sum()
mnm_df = new_df[(new_df['blockType'] == 'MnM')]
mnm_df = mnm_df.groupby('subj').sum() 

combine_df = combine_df.reset_index()
combine_df['type'] = 'Maintain + Monitor'
mnm_df = mnm_df.reset_index()
mnm_df['type'] = 'MnM'  

#both_pm = combine_df.pm_cost  
#mnm_pm = mnm_df.pm_cost 
##all_pm_df = pd.concat([combine_df, mnm_df], axis=0, sort=False)
all_pm_df = pd.concat([combine_df, mnm_df], axis=0)
#all_pm_df.columns = 'maintain_monitor', 'mnm' 
#all_pm_df = all_pm_df.reset_index()   

both_pm = combine_df.pm_cost  
mnm_pm = mnm_df.pm_cost 
##all_2 = pd.concat([both_pm, mnm_pm], axis=1, sort=False)
all_2 = pd.concat([both_pm, mnm_pm], axis=1)
all_2.columns = 'maintain_monitor', 'mnm' 
import pingouin as pg
post_hoc = pg.ttest(all_2.maintain_monitor, all_2.mnm, paired=True)

#remove s18 because they don't have a pm cost
all_pm_df = all_pm_df[(all_pm_df['subj'] != 's18')] 



#all_pm_df[(all_pm_df['subj'] == 's01')].plot(kind="bar") 

for i, j in all_pm_df.iterrows():  
	sea.relplot(x='type', y='pm_cost', data = all_pm_df[(all_pm_df['subj'] == j.subj)], kind = "line") 
	sea.barplot(x='type', y = 'pm_cost', data = all_pm_df[(all_pm_df['subj'] == j.subj)], palette = "Purples")
	#all_pm_df[(all_pm_df['subj'] == j.subj)].plot(color = 'purple',kind="bar") 
	#plt.colorPalette(my_pal)
	#plt.xlabel('Block type')
	#plt.ylabel('PM cost')
	#plt.savefig(FIGURE_PATH + 'allSubj_pmCompare' + j.subj + '.eps', dpi = 600)
	#plt.close()
	#plt.legend('Maintain + Monitor', 'MnM')

sea.pointplot(x = 'type', y = 'pm_cost', hue = 'subj', data = all_pm_df)  

ax = sea.pointplot(x = 'type', y = 'pm_cost', hue = 'subj', data = all_pm_df)
plt.xlabel('Block type')
plt.ylabel('PM cost')
plt.savefig(FIGURE_PATH + 'pm_compare.eps', dpi = 600)


fig, ax = plt.subplots() 
for i, j in all_pm_df.iterrows():
	sea.pointplot(x='type', y='pm_cost', data = all_pm_df[(all_pm_df['subj' == j.subj)], ax = ax, kind = "line") 
