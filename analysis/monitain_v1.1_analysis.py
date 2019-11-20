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

# Create a dataframe from these subjects
df_list = []
for i, fn in enumerate(fnames): 
	subj_df = pd.read_csv(fn, index_col=0)
	# Add subj and trial num to df
	subj_df['subj'] = 's{:02d}'.format(i+1)
	subj_df['trial'] = range(subj_df.shape[0])

	# Indicate what version of the expt subj completed
	if 'v1.0' in fn: 
		vers = 'vers1.0'
	elif 'v1.1' in fn: 
		vers = 'vers1.1'

	subj_df['version'] = vers

	## Code for fixing if num_n was recorded instead of n
	## Not currently used because their data s28 was the only one that repeatedly had
	## the wrong thing recorded 
	## s28 removed for consistency in that everyone was doing the same thing
	# # Change if they hit num_1 to 1 and num_2 to 2
	# # This is a bug in sub 28 (and maybe more) 
	# for trial in subj_df.index: 
	# 	for probe in range(0,15):
	# 		word_or_non = subj_df.iloc[trial, subj_df.columns.get_loc('word{:d}_cond'.format(probe))]
	# 		value = subj_df.loc[trial, 'respProbe{:d}'.format(probe)]
	# 		if (isinstance(value, str) and 'num_1' in value) and (word_or_non == 'word'): 
	# 			print ('subj ', i)
	# 			#print ('trial', trial)
	# 			subj_df.iloc[trial, subj_df.columns.get_loc('probe{:d}_acc'.format(probe))] = 1 ## correct - picked word for word
	# 		elif (isinstance(value, str) and 'num_1' in value) and (word_or_non != 'word'):
	# 			print ('subj ', i)
	# 			#print ('trial', trial)
	# 			subj_df.iloc[trial, subj_df.columns.get_loc('probe{:d}_acc'.format(probe))] = 0 ## incorrect - picked word for nonword
	# 		elif (isinstance(value, str) and 'num_2' in value) and (word_or_non == 'nonword'):
	# 			print ('subj ', i)
	# 			#print ('trial', trial)
	# 			subj_df.iloc[i, subj_df.columns.get_loc('probe{:d}_acc'.format(probe))] = 1 ## correct - picked nonword for nonword
	# 		elif (isinstance(value, str) and 'num_2' in value) and (word_or_non != 'nonword'): 
	# 			print ('subj ', i)
	# 			#print ('trial', trial)
	# 			subj_df.iloc[trial, subj_df.columns.get_loc('probe{:d}_acc'.format(probe))] = 0 ## incorrect - picked word for nonword
	# 		elif (isinstance(value, str) and 'num' in value): 
	# 			print ('subj ', i)
	# 			print ('trial', trial)
	# 			print ('value', value)

	subj_df = subj_df[1:] #get rid of first line because something is off
	df_list.append(subj_df)

##df_main = pd.concat(df_list,ignore_index=False, sort=False)
df_main = pd.concat(df_list,ignore_index=False)

# Add new column for RT on PM probes
df_main['pm_probe_rt'] = np.nan

df_main = df_main.reset_index()

# Make a copy to replace inaccurate responses with no RT
df_main_copy = df_main
# Use for backup



# Master list of all rt probe titles
rtProbes = []
for rt in range(0,14): #get rid of probe14 bc you'll never look at it 
	rtProbes.append('rtProbe{:d}'.format(rt))

accCols = []
for acc in range(0,15): #get rid of probe14 bc you'll never look at it 
	accCols.append('probe{:d}_acc'.format(acc))



# Record PM accuracy per trial
baseline_str = "base"
for trial in df_main.index: 
	probeNum = df_main.loc[trial, 'n_probes']
	# Move RT for PM probe to it's own column and then replace probeX with nan so it 
	# doesn't get averaged in with PM cost later
	index_base = (df_main.loc[trial, 'block']).find(baseline_str)
	if index_base == -1: 
		df_main.loc[trial, 'pm_probe_rt'] = df_main.loc[trial, 'rtProbe{:d}'.format(probeNum-1)]
		df_main.loc[trial, 'rtProbe{:d}'.format(probeNum-1)] = np.nan;

	if df_main.loc[trial, 'index'] == 0: #something wrong with first trial 
		pass
	elif df_main.loc[trial, 'probe{:d}_acc'.format(probeNum-1)] == 1: #correct PM resp
		df_main.loc[trial, 'acc'] = 1
		# add rt from PM probe into PM RT probe column and then set this to NaN so it doesn't average in for PM cost
		# note that this will add the 1 RT for baseline probes to pm probe column so don't do that for baseline
		
	else:
		df_main.loc[trial, 'acc'] = 0




################################
####### Excluding data #########
################################

#Replace RT with nan if OG task was incorrect
#Replace RT with nan in resp was quicker than 0.05 secs, also replace pm acc with nan
#Cut off based on findings from "Eye Movements and Visual Encoding during Scene Perception"
#Rayner et al., 2009
 
for i in df_main.index: 
	if df_main.loc[i, 'pm_probe_rt'] < 0.05: 
		df_main.at[i, 'pm_probe_rt'] = np.nan

	for probe in range(0,15): 
		# Replace rt with nan if probe resp was incorrect
		if df_main.loc[i, 'probe{:d}_acc'.format(probe)] == 0: 
			df_main.at[i, 'rtProbe{:d}'.format(probe)] = np.nan
		# Replace rt and acc with nan if probe RT was quicker than 0.05 secs
		if df_main.loc[i, 'rtProbe{:d}'.format(probe)] < 0.05: 			
			df_main.at[trial, 'rtProbe{:d}'.format(probe)] = np.nan
			df_main.at[trial, 'probe{:d}_acc'.format(probe)] = np.nan


#remove s18 because they don't have a pm cost
#remove s28 because they didn't follow the instructions
df_main = df_main[(df_main['subj'] != 's18')] 
df_main = df_main[(df_main['subj'] != 's28')] 
#all_pm_df = all_pm_df[(all_pm_df['subj'] != 's28')]  




################################
### By block type dataframes ###
################################


def createBlockDFs(str1, str2, blockType):
	block_name = df_main[(df_main['block'] == str1) | (df_main['block'] == str2)]	
	##block_name_df = pd.concat([ block_name['subj'], block_name['block'], block_name['acc'], block_name[rtProbes].mean(axis=1), block_name[accCols].mean(axis=1)], axis=1 , sort=False)
	block_name_df = pd.concat([ block_name['subj'], block_name['block'], block_name['acc'], block_name['pm_probe_rt'],
		block_name[rtProbes].mean(axis=1), block_name[accCols].mean(axis=1)], axis=1)
	block_name_df['blockType'] = blockType
	block_name_df.columns = ['subj', 'block', 'pm_acc', 'pm_probe_rt', 'meanTrial_rt', 'og_acc', 'blockType',]	
	return block_name_df

block_baseline_df = createBlockDFs("('base1', 1)", "('base2', 8)", "Baseline")
block_maintain_df = createBlockDFs("('maintain1', 2)", "('maintain2', 5)", "Maintain")
block_monitor_df = createBlockDFs("('monitor1', 3)", "('monitor2', 6)", "Monitor")
block_mnm_df = createBlockDFs("('mnm1', 4)", "('mnm2', 7)", "MnM")




################################
###### Calculate PM cost #######
################################

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




################################
#### All blocks dataframe ######
################################

##all_df = pd.concat([block_baseline_df, block_maintain_df, block_monitor_df, block_mnm_df], axis = 0, sort=False)
##all_df_minusBase = pd.concat([block_maintain_df, block_monitor_df, block_mnm_df], axis = 0, sort=False)
all_df = pd.concat([block_baseline_df, block_maintain_df, block_monitor_df, block_mnm_df], axis = 0)
all_df_minusBase = pd.concat([block_maintain_df, block_monitor_df, block_mnm_df], axis = 0)
# If just looking at all_df_minusBase, need to redo exclusion criteria for that




################################
#### Dataframes for MnM & ######
##### Maintain + Monitor #######
################################

# Combine everything
new_df = all_df.groupby(['subj','blockType']).mean().reset_index(drop=False)

# Make a dataframe of just Maintain blocks and Monitor trials
combine_df = new_df[(new_df['blockType'] == 'Monitor') |(new_df['blockType']=='Maintain')]
# Add maintain plus monitor for each subject
combine_df = combine_df.groupby('subj').sum()
combine_df = combine_df.reset_index()
combine_df['type'] = 'Maintain + Monitor'

# Make a dataframe of just MnM trials
mnm_df = new_df[(new_df['blockType'] == 'MnM')]
mnm_df = mnm_df.groupby('subj').sum() 
mnm_df = mnm_df.reset_index()
mnm_df['type'] = 'MnM'  

# Make a dataframe of with a column for Maintain + Monitor alongside column
# for MnM for each subject
#both_pm = combine_df.pm_cost  
#mnm_pm = mnm_df.pm_cost 
##all_pm_df = pd.concat([combine_df, mnm_df], axis=0, sort=False)
all_pm_df = pd.concat([combine_df, mnm_df], axis=0)
#all_pm_df.columns = 'maintain_monitor', 'mnm' 
#all_pm_df = all_pm_df.reset_index() 

# Pull out PM cost
both_pm = combine_df.pm_cost  
mnm_pm = mnm_df.pm_cost 
##all_2 = pd.concat([both_pm, mnm_pm], axis=1, sort=False)

# Make a dataframe of just PM cost
all_2 = pd.concat([both_pm, mnm_pm], axis=1)
all_2.columns = 'maintain_monitor', 'mnm' 




################################
##### By subject figures #######
################################

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




################################
######### Functions to #########
######## create figures ########
################################

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
	plt.ylabel('RT cost (s)')
	sea.despine()
	#plt.title('Average PM cost by block type')
	plt.savefig(FIGURE_PATH + 'allSubj_PMCOST.eps', dpi = 600)
	plt.close()

def allSubj_pmCompare_point():
	## Point plot of pmCost of maintain + pmCost of monitor compared to pmCost of MnM
	## All subjects on one plot
	fig, ax = plt.subplots() 
	for i, j in all_pm_df.iterrows():
		sea.pointplot(x = 'type', y = 'pm_cost', data = all_pm_df[(all_pm_df['subj'] == j.subj)], color = '0.75', scale = 0.7, ax = ax)
	plt.xlabel('Block type');
	plt.ylabel('PM cost');
	plt.savefig(FIGURE_PATH + 'allSubj_pmCompare_point.eps', dpi = 600)
	plt.close()


def allSubj_pmCompare_pointPlusViolin():
	## Point plot of pmCost of maintain + pmCost of monitor compared to pmCost of MnM
	## All subjects on one plot    
	fig, ax = plt.subplots() 
	for i, j in all_pm_df.iterrows():
		sea.pointplot(x = 'type', y = 'pm_cost', data = all_pm_df[(all_pm_df['subj'] == j.subj)], color = '0.25', scale = 0.5, ax = ax)
	ax = sea.violinplot(x = all_pm_df.type, y = all_pm_df.pm_cost, palette=['palevioletred', 'mediumpurple'])
	plt.xlabel('Block type');
	plt.ylabel('PM cost');
	plt.savefig(FIGURE_PATH + 'allSubj_pmCompare_point_violin.eps', dpi = 600)
	plt.close()




################################
######### Figures for ##########
######### all subjects #########
################################

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

allSubj_pmAcc()
allSubj_ogAcc()
allSubj_rt()
allSubj_pmCost()
allSubj_pmCompare_point()
allSubj_pmCompare_pointPlusViolin()

maintain_val = []
monitor_val = []
pm_acc_val = []
smaller = all_df.groupby(['subj', 'blockType']).mean() 
for index, row in smaller.iterrows(): 
	if (index[1] == 'Maintain'):
		maintain_val.append(row.pm_cost)
	elif (index[1] == 'Monitor'):
		monitor_val.append(row.pm_cost)
	elif (index[1] == 'MnM'): 
		pm_acc_val.append(row.pm_acc)

result = np.subtract(maintain_val, monitor_val)   

sea.scatterplot(x = result, y = pm_acc_val) 
plt.axvline(x = 0, linestyle='--', color = 'gray')
plt.xlabel('Maintain cost - Monitor cost')
plt.ylabel('Average PM accuracy on combined trials')
plt.savefig(FIGURE_PATH + 'maintain_v_monitor.png', dpi = 600)
plt.close()




################################
######### Export CSVs ##########
################################

#### Export dataframes to CSVs
CSV_PATH = FIGURE_PATH + 'csvs'

## All DFs
all_df_averaged = all_df.groupby(['subj', 'blockType']).mean().reset_index(drop = False)
pmCost_df_averaged = pmCost_df.groupby(['subj', 'blockType']).mean().reset_index(drop = False)

fname_all = os.path.join(CSV_PATH, 'ALL.csv')
all_df_averaged.to_csv(fname_all, index = False)
fname_pm = os.path.join(CSV_PATH, 'PM_COST.csv')
pmCost_df_averaged.to_csv(fname_pm, index = False)

fname_all_byTrial = os.path.join(CSV_PATH, 'ALL_BYTRIAL.csv')
all_df.to_csv(fname_all_byTrial, index = False)




