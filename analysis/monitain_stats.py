##################################
##################################
######## monitain STATS ##########
########### for v.1.1 ############
######## Katie Hedgpeth ##########
############ updated #############
######### November 2019 ##########
##################################

import os
import pingouin as pg
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
import math

from scipy.stats import pearsonr

#from sklearn.linear_model import LinearRegression

#import bootstrapped.bootstrap as baseline
#import bootstrapped.stats_functions as bs_stats

################################
##### Import data & setup ######
################################

## Set paths for where files will be opened from or saved
PATH = os.path.expanduser('~')
FIGURE_PATH = PATH + '/monitain/analysis/output/'
CSV_PATH = FIGURE_PATH + 'csvs'

# Read in CSVs
all_df_averaged = pd.read_csv(FIGURE_PATH+'/csvs/ALL.csv')
pmCost_df_averaged = pd.read_csv(FIGURE_PATH+'/csvs/PM_COST.csv')
all_df_byTrial = pd.read_csv(FIGURE_PATH+'csvs/ALL_BYTRIAL.csv')

# Remove baseline trials (blocks 1 and 8)
# Add this back in (can be done by commenting out line below, to plot baseline)
all_df_averaged_minusBase = all_df_averaged[all_df_averaged.blockType != 'Baseline']   

# If you needed to remove a specific subject, do this and change 's18'
##all_df_averaged = all_df_averaged[(all_df_averaged['subj'] != 's18')]




################################
##### ANOVA and Post-hoc #######
################################

## og acc
aov_og = pg.rm_anova(dv='og_acc', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_og = pg.pairwise_ttests(dv='og_acc', within='blockType', data=all_df_averaged, padjust='bonferroni')

## pm acc
aov_pm = pg.rm_anova(dv='pm_acc', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_pm = pg.pairwise_ttests(dv='pm_acc', within='blockType', subject = 'subj', data=all_df_averaged, padjust='bonferroni')

## rt
aov_rt = pg.rm_anova(dv='meanTrial_rt', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_rt = pg.pairwise_ttests(dv='meanTrial_rt', within='blockType', subject = 'subj', data=all_df_averaged, padjust='bonferroni')

## pm cost
aov_pmCost = pg.rm_anova(dv='pm_cost', within = 'blockType', subject = 'subj', data=pmCost_df_averaged, detailed = True)
posthoc_pmCost = pg.pairwise_ttests(dv='pm_cost', within='blockType', subject = 'subj', data=pmCost_df_averaged, padjust='bonferroni')


### export p values to dataframe and then csv

# utility functions
def findSignificance(p_val_list, sig_list):
	for i in range(len(p_val_list)): 
		if p_val_list[i] < 0.001: 
			sig_list.append('***')
		elif p_val_list[i] < 0.01: 
			sig_list.append('**')
		elif p_val_list[i] < 0.05: 
			sig_list.append('*')
		elif p_val_list[i] < 0.1: 
			sig_list.append('.')
		else: 
			sig_list.append(' ')

def addToList(posthoc_type):
	for i in range(len(posthoc_type)): 
		p_val_list_ttests.append(posthoc_type['p-corr'][i])
		condition_list_ttests.append(posthoc_type['A'][i] + " + " + posthoc_type['B'][i])

# set index
data_indx_anova = ['og_acc', 'pm_acc', 'rt', 'pmCost']



### RM ANOVAS

# create lists for anovas
p_val_list_anova = [aov_og['p-unc'][0], aov_pm['p-unc'][0],
	aov_rt['p-unc'][0], aov_pmCost['p-unc'][0]]
sig_list_anova = []

# find sig codes for anova
findSignificance(p_val_list_anova, sig_list_anova)

## create df for anova data
anova_data = {'p-unc': p_val_list_anova, 'sig code': sig_list_anova}
rm_anova_df = pd.DataFrame(anova_data, index = data_indx_anova)
fname_anova = os.path.join('rm_anova_pvals.csv')
rm_anova_df.to_csv(fname_anova)



### POST-HOC T-TESTs

# set index
data_indx_ttest = ([('og_acc')] * len(posthoc_og))  + ([('pm_acc')] * len(posthoc_pm)) + ([('rt')] * len(posthoc_rt))  + ([('pmCost')] * len(posthoc_pmCost))
 
# create empty lists for ttests
p_val_list_ttests = []
condition_list_ttests = []
sig_list_ttests = []

# fill in lists
addToList(posthoc_og)
addToList(posthoc_pm)
addToList(posthoc_rt)
addToList(posthoc_pmCost)

# find sig codes for ttest results
findSignificance(p_val_list_ttests, sig_list_ttests)

## create df for ttest data
ttest_data = {'p-corr': p_val_list_ttests, 'condition': condition_list_ttests, 'sig code': sig_list_ttests}
ttest_df = pd.DataFrame(ttest_data, index = data_indx_ttest)
# export to csv
fname_ttest = os.path.join('ttest_pvals.csv')
ttest_df.to_csv(fname_ttest)




### LOGISTIC REGRESSION

##### BY SUBJECT 

#X = all_df_byTrial.pm_cost
#y = all_df_byTrial.pm_acc
#pg.logistic_regression(X, y, remove_na=True)  

## create scatter plot of x = pm cost and y = pm acc
# ax = sea.barplot(x="pm_acc", y = "pm_cost", data=all_df_byTrial) 
# plt.xlabel('PM accuracy')
# plt.ylabel('PM cost (secs)')
# plt.savefig(FIGURE_PATH + 'pmAcc_v_pmCost.png', dpi = 600)
# plt.close()

############


################################
###### Create dataframes #######
######## of all trials #########
################################

# dfs of alllll trials by block type
maintain_all = all_df_byTrial[(all_df_byTrial['blockType'] == 'Maintain')]
maintain_all = maintain_all.drop(columns=['block'])

monitor_all = all_df_byTrial[(all_df_byTrial['blockType'] == 'Monitor')]
monitor_all = monitor_all.drop(columns=['block'])

mnm_all = all_df_byTrial[(all_df_byTrial['blockType'] == 'MnM')]
mnm_all = mnm_all.drop(columns=['block'])

cost_columns_drop = ['blockType', 'pm_acc', 'meanTrial_rt','og_acc', 'pm_probe_rt']
acc_columns_drop = ['blockType','pm_cost', 'meanTrial_rt','og_acc', 'pm_probe_rt']

maintain_cost_all = maintain_all.drop(columns=cost_columns_drop, axis=1).reset_index(drop=True)
maintain_acc_all = maintain_all.drop(columns=acc_columns_drop, axis=1).reset_index(drop=True)

monitor_cost_all = monitor_all.drop(columns=cost_columns_drop, axis=1).reset_index(drop=True)
monitor_acc_all = monitor_all.drop(columns=acc_columns_drop, axis=1).reset_index(drop=True)

mnm_cost_all = mnm_all.drop(columns=cost_columns_drop, axis=1).reset_index(drop=True)
mnm_acc_all = mnm_all.drop(columns=acc_columns_drop, axis=1).reset_index(drop=True)




def matchingSubjs(df1, df2):
	assert len(df1) == len(df2)
	for row, index in df1.iterrows():
		if df1.loc[row].subj != df2.loc[row].subj:
			print("the subjects do not match up in the two dataframes")
	df2_drop = df2.drop(columns = ['subj'])
	return df2_drop	


################################
##### Combine dataframes #######
################################

# 2s added because subj eliminated after running through maintainSubjs in second df
# want to keep that to double check s01 is combined with s01 and so on
# Maintain cost, maintain accuracy
maintain_acc_all2 = matchingSubjs(maintain_cost_all, maintain_acc_all)
maintainCost_maintainAcc_all = pd.concat([maintain_cost_all, maintain_acc_all2], axis = 1)

# Monitor cost, monitor accuracy
monitor_acc_all2 = matchingSubjs(monitor_cost_all, monitor_acc_all)
monitorCost_monitorAcc_all = pd.concat([monitor_cost_all, monitor_acc_all2], axis = 1)

# MnM cost, MnM accuracy
mnm_acc_all2 = matchingSubjs(mnm_cost_all, mnm_acc_all)
mnmCost_mnmAcc_all = pd.concat([mnm_cost_all, mnm_acc_all2], axis = 1)

# Maintain cost, MnM accuracy
mnm_acc_all2 = matchingSubjs(maintain_cost_all, mnm_acc_all)
maintainCost_mnmAcc_all = pd.concat([maintain_cost_all, mnm_acc_all2], axis = 1)

# Monitor cost, MnM accuracy
mnm_acc_all2 = matchingSubjs(monitor_cost_all, mnm_acc_all)
monitorCost_mnmAcc_all = pd.concat([monitor_cost_all, mnm_acc_all2], axis = 1)




################################
###### Create dataframes #######
###### by block averages #######
################################

cost_all_dropColumns = ['pm_acc', 'meanTrial_rt','og_acc', 'pm_probe_rt']
result_all_dropColumns = ['subj', 'pm_cost', 'meanTrial_rt','og_acc', 'pm_probe_rt']
pmRT_all_dropColumns = ['subj', 'pm_cost', 'meanTrial_rt', 'og_acc', 'pm_acc']

# Does maintenance cost predict combined performance? 
maintain_trials = all_df_averaged[(all_df_averaged['blockType'] == 'Maintain')] # maintain block trials
monitor_trials = all_df_averaged[(all_df_averaged['blockType'] == 'Monitor')] # monitor block trials
mnm_trials = all_df_averaged[(all_df_averaged['blockType'] == 'MnM')] # mnm block trials

maintain_results = maintain_trials.groupby(['subj']).mean().reset_index()
maintain_cost = maintain_results.drop(columns=cost_all_dropColumns, axis=1)
maintain_acc = maintain_results.drop(columns=result_all_dropColumns , axis=1) 
maintain_pmRT = maintain_results.drop(columns=pmRT_all_dropColumns, axis = 1)
# Remove 'subj' column if you need a subj num indication in accuracy df

monitor_results = monitor_trials.groupby(['subj']).mean().reset_index()
monitor_cost = monitor_results.drop(columns=cost_all_dropColumns, axis=1)
monitor_acc = monitor_results.drop(columns=result_all_dropColumns , axis=1) 
monitor_pmRT = monitor_results.drop(columns=pmRT_all_dropColumns, axis = 1)

mnm_results = mnm_trials.groupby(['subj']).mean().reset_index()
mnm_cost = mnm_results.drop(columns=cost_all_dropColumns, axis=1)
mnm_acc = mnm_results.drop(columns=result_all_dropColumns , axis=1) 
mnm_pmRT = maintain_results.drop(columns=pmRT_all_dropColumns, axis = 1)




################################
##### Combine dataframes #######
################################

# Maintain cost versus combined performance
mainCost_combineAcc = pd.concat([maintain_cost, mnm_acc], axis = 1, sort = False)
# Monitor cost versus combined performance
monCost_combineAcc = pd.concat([monitor_cost, mnm_acc], axis = 1, sort = False)
# Combined cost versus combined performance
combineCost_combineAcc = pd.concat([mnm_cost, mnm_acc], axis = 1, sort = False)

# Maintain cost versus maintain performance
mainCost_mainAcc = pd.concat([maintain_cost, maintain_acc], axis = 1, sort = False)
# Monitor cost versus monitor performance
monCost_monAcc = pd.concat([monitor_cost, monitor_acc], axis = 1, sort = False)

# Maintain cost + Monitor cost
combined_cost = maintain_cost.pm_cost + monitor_cost.pm_cost
# Combined cost (above) versus combined performance
mplusmCost_combineAcc = pd.concat([combined_cost, mnm_acc], axis=1, sort=False) 



# Add somewhere: 
all_df_short = all_df_minusBase.drop(columns = ['block', 'meanTrial_rt', 'og_acc']).reset_index(drop=True) 
all_df_short['main_cost'] = cost_acc_df.main_cost 



################################
###### Create dataframes #######
####### with everything ########
################################

## Create a df with everything - for each subj, a column for maintain cost, monitor cost, mnm cost, maintain acc, etc.
mainCost = maintain_cost.rename(columns = {"pm_cost" : "main_cost"})
monCost = monitor_cost.rename(columns = {"pm_cost" : "mon_cost"}).drop(columns=['subj'])
mnmCost = mnm_cost.rename(columns = {"pm_cost" : "mnm_cost"}).drop(columns=['subj'])

mainAcc = maintain_acc.rename(columns = {"pm_acc":"main_acc"}) 
monAcc = monitor_acc.rename(columns = {"pm_acc":"mon_acc"}) 
mnmAcc = mnm_acc.rename(columns = {"pm_acc":"mnm_acc"}) 

cost_acc_df = pd.concat([mainCost, monCost, mnmCost, mainAcc, monAcc, mnmAcc], axis=1)
# Save to CSV for R
fname_df = os.path.join(CSV_PATH, 'cost_acc.csv')
cost_acc_df.to_csv(fname_df, index = False)




################################
########## Drop NaNs ###########
################################

mainCost_combineAcc = mainCost_combineAcc.dropna()
monCost_combineAcc = monCost_combineAcc.dropna()
combineCost_combineAcc = combineCost_combineAcc.dropna()

mainCost_mainAcc = mainCost_mainAcc.dropna()
monCost_monAcc = monCost_monAcc.dropna()

mplusmCost_combineAcc = mplusmCost_combineAcc.dropna()




################################
######### Correlations #########
################################

pearsonr(cost_acc_df.main_cost, cost_acc_df.main_acc) 
pearsonr(cost_acc_df.mon_cost, cost_acc_df.mon_acc) 

pearsonr(cost_acc_df.main_cost, cost_acc_df.mnm_cost) 
pearsonr(cost_acc_df.mon_cost, cost_acc_df.mnm_cost) 

pearsonr(cost_acc_df.main_cost, cost_acc_df.mnm_acc) 
pearsonr(cost_acc_df.mon_cost, cost_acc_df.mnm_acc) 
pearsonr(cost_acc_df.mnm_cost, cost_acc_df.mnm_acc) 

ax = sea.scatterplot(x = 'main_cost', y = 'mon_cost', data = cost_acc_df, hue= 'mnm_acc', size = 'mnm_acc', sizes = (20, 200))
ax.set_xlabel('Maintain cost')
ax.set_ylabel('Monitor cost')  
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles = handles[1:], labels = labels[1:])
ax.legend(handles=handles[1:], labels=['0.3', '0.6', '0.9'], title="PM accuracy", loc = 'lower right')
plt.savefig(FIGURE_PATH + 'mainCost_v_monCost.eps', dpi = 600)
plt.close()


################################
########## Regression ##########
################################

main_main = maintainCost_maintainAcc_all.dropna()
mon_mon = monitorCost_monitorAcc_all.dropna()

def regPlot(combinedData, x_label, y_label, color, y_data, xLimit, yLimit): 
	fig, ax = plt.subplots()
	ax.set_xlim(xLimit)
	ax.set_ylim(yLimit)	
	ax = sea.regplot(x='pm_cost', y = y_data, data = combinedData, color = color, scatter = True) 
	plt.xlabel(x_label)
	plt.ylabel(y_label)



def residPlot(combinedData, x_label, y_label, color):
	sea.residplot(x = 'pm_cost', y = 'pm_acc', data = main_V_main, color = color, scatter_kws = {"s": 80}) 
	plt.xlabel(x_label)
	plt.ylabel(y_label)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sea.regplot(x = 'pm_cost', y = 'pm_acc', data = main_main, color = 'b', logistic=True, ax = ax1)
ax1.set_xlabel('Maintain cost')
ax1.set_ylabel('Maintain performance') 

sea.regplot(x = 'pm_cost', y = 'pm_acc', data = mon_mon, color = 'r', logistic=True, ax = ax2)
ax2.set_xlabel('Monitor cost')
ax2.set_ylabel('Monitor performance') 

plt.savefig(FIGURE_PATH + 'logreg_compare.png', dpi = 600)
plt.close()




##TODO: Change file extension based on what the image is for
# png for Slack
# pdf for viewing
# eps for editing


### Cost v accuracy
## How does maintainance cost affect performance when only maintaining?
regPlot(mainCost_mainAcc, 'Maintain cost (s)', 'Maintain performance', 'b', 'pm_acc', [-0.2,0.6], [-.1,1])

plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainPerf.eps', dpi = 600)
plt.close()

maintain_maintain_lr = pg.linear_regression(mainCost_mainAcc.pm_cost, mainCost_mainAcc.pm_acc) 

### Cost v RT
#regPlot(mainCost_mainRT, 'Maintain cost (s)', 'Maintain PM probe RT', 'b', 'pm_probe_rt')
#plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainRT.png', dpi = 600)
#plt.close()

#maintain_maintain_RT_lr = pg.linear_regression(mainCost_mainRT.pm_cost, mainCost_mainRT.pm_probe_rt)


### Cost v accuracy
## How does monitoring cost affect performance when only monitoring?
regPlot(monCost_monAcc, 'Monitor cost (s)', 'Monitor performance', 'r', 'pm_acc', [-0.2,0.6], [-.1,1])
plt.savefig(FIGURE_PATH + 'monitorCost_v_monitorPerf.png', dpi = 600)
plt.close()

monitor_monitor_lr = pg.linear_regression(monCost_monAcc.pm_cost, monCost_monAcc.pm_acc)

### Cost v RT
#regPlot(monCost_monRT, 'Monitor cost (s)', 'Monitor PM probe RT', 'r', 'pm_probe_rt')
#plt.savefig(FIGURE_PATH + 'monitorCost_v_monitorRT.png', dpi = 600)
#plt.close()

#monitor_monitor_RT_lr = pg.linear_regression(monCost_monRT.pm_cost, monCost_monRT.pm_probe_rt)



### Cost v accuracy
## How does maintaining cost affect PM performance?
regPlot(mainCost_combineAcc, 'Maintain cost (s)','Combined performance', 'b', 'pm_acc', [-0.2,0.6], [-.1,1])
plt.savefig(FIGURE_PATH + 'maintainCost_v_pmAcc.png', dpi = 600)
plt.close()

maintain_combine_lr = pg.linear_regression(mainCost_combineAcc.pm_cost, mainCost_combineAcc.pm_acc)

### Cost v RT
#regPlot(mainCost_combineRT, 'Maintain cost (s)','Combined PM probe RT', 'b', 'pm_probe_rt')
#plt.savefig(FIGURE_PATH + 'maintainCost_v_pmRT.png', dpi = 600)
#plt.close()

#maintain_combine_RT_lr = pg.linear_regression(mainCost_combineRT.pm_cost, mainCost_combineRT.pm_probe_rt)


### Cost v accuracy
## How does monitoring cost affect PM performance?
regPlot(monCost_combineAcc, 'Monitor cost (s)','Combined performance', 'r', 'pm_acc', [-0.2,0.6], [-.1,1])
plt.savefig(FIGURE_PATH + 'monitorCost_v_pmAcc.png', dpi = 600)
plt.close()

monitor_combine_lr = pg.linear_regression(monCost_combineAcc.pm_cost, monCost_combineAcc.pm_acc)

### Cost v RT
#regPlot(monCost_combineRT, 'Monitor cost (s)','Combined PM probe RT', 'r', 'pm_probe_rt')
#plt.savefig(FIGURE_PATH + 'monitorCost_v_pmRT.png', dpi = 600)
#plt.close()

#monitor_combine_RT_lr = pg.linear_regression(monCost_combineRT.pm_cost, monCost_combineRT.pm_probe_rt)


### Cost v accuracy
## How does PM cost affect performance when maintaining AND monitoring?
regPlot(combineCost_combineAcc, 'Combined cost (s)','Combined performance', 'purple', 'pm_acc',[-0.2,0.6], [-.1,1])
plt.savefig(FIGURE_PATH + 'mnmCost_v_pmAcc.png', dpi = 600)
plt.close()

combine_combine_lr = pg.linear_regression(combineCost_combineAcc.pm_cost, combineCost_combineAcc.pm_acc)

# Plot all three of maintain, monitor, and combined on top of one another
# Set Scatter = False in function if you don't want scatter points
regPlot(combineCost_combineAcc, 'Combined cost (s)','Combined performance', 'purple', 'pm_acc',[-0.2,0.6], [-.1,1])
regPlot(monCost_combineAcc, 'Monitor cost (s)','Combined performance', 'r', 'pm_acc', [-0.2,0.6], [-.1,1])
regPlot(mainCost_combineAcc, 'Reaction time cost (s)','Combined performance', 'b', 'pm_acc', [-0.2,0.6], [-.1,1])

fig, ax = plt.subplots()
ax.set_xlim([-0.2,0.6])
ax.set_ylim([-.1,1])
ax = sea.regplot(x='pm_cost', y = 'pm_acc', data = combineCost_combineAcc, color = 'purple', scatter = False) 
ax = sea.regplot(x='pm_cost', y = 'pm_acc', data = monCost_combineAcc, color = 'r', scatter = False) 
ax = sea.regplot(x='pm_cost', y = 'pm_acc', data = mainCost_combineAcc, color = 'b', scatter = False) 
plt.xlabel('RT cost (s)')
plt.ylabel('Combined perfomrance')



### Cost v accuracy
## How does maintenance cost + monitoring cost relate to performance when maintaining AND monitoring?
regPlot(mplusmCost_combineAcc, 'Maintainence + Monitoring cost (s)','Combined performance', 'mediumvioletred', 'pm_acc', [-0.2,0.5], [0,1])
plt.savefig(FIGURE_PATH + 'mplusmCost_v_combine_pmAcc.png', dpi = 600)
plt.close()

mplusm_combine_lr = pg.linear_regression(mplusmCost_combineAcc.pm_cost, mplusmCost_combineAcc.pm_acc)

### Cost v cost
## How does maintenance cost + monitoring cost relate to MnM cost?
mplusmCost_mnmCost = pd.concat([combined_cost, mnmCost], axis=1, sort=False)
mplusmCost_mnmCost = mplusmCost_mnmCost.rename(columns = {"pm_cost":"mplusm_cost"}) 

fig, ax = plt.subplots() 
ax.set_xlim([-0.2,0.6])
ax.set_ylim([-0.2,0.6])
ax = sea.regplot(x='mplusm_cost', y = 'mnm_cost', data = mplusmCost_mnmCost, color = 'mediumvioletred', scatter = True) 
plt.xlabel('Maintainence + Monitoring cost (s)')
plt.ylabel('Combined cost (s)')
plt.savefig(FIGURE_PATH + 'mplusmCost_v_combineCost.png', dpi = 600)
plt.close()

cost_mplusm_combine_lr = pg.linear_regression(mplusmCost_mnmCost.mplusm_cost, mplusmCost_mnmCost.mnm_cost )


regPlot(mplusmCost_mnmCost, 'RT cost (s)','Combined performance', 'purple', 'pm_acc', [-0.2,0.6], [0,1])
plt.savefig(FIGURE_PATH + 'mplusmCost_v_combine_pmAcc.png', dpi = 600)
plt.close()

fig, ax = plt.subplots()
ax.set_xlim([-0.2,0.6])
ax.set_ylim([0,1])	
ax = sea.regplot(x='pm_cost', y = 'pm_acc', data = mplusmCost_combineAcc, color = 'palevioletred', scatter = True) 
plt.xlabel('Maintain cost + Monitor cost (s)')
plt.ylabel('Combined performance')
plt.savefig(FIGURE_PATH + 'mplusmCost_v_combine_pmAcc.png', dpi = 600)
plt.close()

mplusm_combine_lr = pg.linear_regression(mplusmCost_combineAcc.pm_cost, mplusmCost_combineAcc.pm_acc)



# Plot Maintain + Monitor on top of MnM
# Set scatter to false
f, ax = plt.subplots() 
ax.set(xlim=[-0.2,0.55], ylim = [0,1])
regPlot(mplusmCost_combineAcc, 'Maintainence + Monitoring cost (s)','Combined performance', 'mediumvioletred', 'pm_acc', [-0.2,0.6], [0,1])
regPlot(combineCost_combineAcc, 'RT cost (s)','Combined performance', 'purple', 'pm_acc',[-0.2,0.6], [0,1])
plt.savefig(FIGURE_PATH + 'mplusmCost_mnm_noScatter.png', dpi = 600)
plt.close()

fig, ax = plt.subplots()
ax.set_xlim([-0.2,0.6])
ax.set_ylim([0,1])	
ax = sea.regplot(x='pm_cost', y = 'pm_acc', data = mplusmCost_combineAcc, color = 'palevioletred', scatter = False) 
ax = sea.regplot(x='pm_cost', y = 'pm_acc', data = combineCost_combineAcc, color = 'purple', scatter = False) 
plt.xlabel('RT cost (s)')
plt.ylabel('Combined performance')




################################
########## Bootstrap ###########
################################

# Array of all of the subjects
subj_list = all_df_byTrial.subj.unique() 

def bootstrapped(trial_df, subj_list, n_iterations, x, y, type, color):

	trial_dict = dict()
	for k, v in trial_df.groupby('subj'):
		trial_dict[k] = v

	# Create dictionary to store bootstrap results
	boot_dict = dict()

	for i in range(n_iterations):
		# Create a new random subject list
		resampled_subjList = np.random.choice(subj_list, size = len(subj_list), replace = True)

		# Create a dictionary for the new resampled subject list
		resampled_dict = dict()

		# For each subject in new subject list, add all of their data
		for subj in resampled_subjList: 
			resampled_dict[subj] = trial_dict.get(subj)

		resampled_df = pd.DataFrame()
		# Merge all of the values from the dictionary together so there is one big dataframe of all new subject data
		for key, value in trial_dict.items(): 
			if value is None: 

				print(key)
		# Ignore index = True so that it resets the index so it goes from 0 to x, rather than random number to random number
		resampled_df = pd.concat(resampled_dict.values(), ignore_index = True)

		# Don't run the below line if you don't care about by subject
		#bootstrap_data = resampled_df.groupby(['subj']).mean()

		#logistic regression, monitor cost by monitor accuracy 
		#bootstrap_lr = pg.linear_regression(bootstrap_data[x], bootstrap_data[y])

		# drop rows where at least one element is missing
		resampled_df = resampled_df.dropna()
		bootstrap_lr = pg.logistic_regression(resampled_df[x], resampled_df[y])

		boot_dict[i] = bootstrap_lr

	# Create df of regression results - include intercept and coefficient for each bootstrap iteration
	betas = pd.DataFrame(columns = ['intercept', 'coef'])
	for key in boot_dict: 
		intercept = boot_dict.get(key).coef[0]
		coef = boot_dict.get(key).coef[1]
		betas.loc[key] = [intercept, coef]


	# Plot betas	
	ax = sea.distplot(betas.coef, color = color) 
	ax.set_xlim([-4,4])
	plt.legend(labels = ['Maintenance betas', 'Monitoring betas'])
	plt.xlabel('Coefficient')
	#plt.savefig(FIGURE_PATH + type + '_bootstrap.eps', dpi = 600)
	#plt.close()

	return betas;


# Bootstrap results
maintain_maintain_betas = bootstrapped(maintainCost_maintainAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_maintain', 'b')
monitor_monitor_betas = bootstrapped(monitorCost_monitorAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_monitor', 'r')


# Don't do this! If it's maintain vs mnm or monitor vs mnm, it won't work
# You can't correlate maintain trial 1 to mnm trial 1, you must take averages
#maintain_maintain_betas = bootstrapped(maintainCost_maintainAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_maintain')
#monitor_monitor_betas = bootstrapped(monitorCost_monitorAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_monitor')
#mnm_mnm_betas = bootstrapped(mnmCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'mnm_mnm')

#maintain_mnm_betas = bootstrapped(maintainCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_mnm')
#monitor_mnm_betas = bootstrapped(monitorCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_mnm')


#maintain_mnm_betas = bootstrapped(mainCost_combineAcc, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_mnm', 'b')
#monitor_mnm_betas = bootstrapped(monCost_combineAcc, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_mnm', 'r')

#maintain_maintain_betas = bootstrapped(mainCost_mainAcc, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_maintain', 'b')
#monitor_monitor_betas = bootstrapped(monCost_monAcc, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_monitor', 'r')
#mnm_mnm_betas = bootstrapped(combineCost_combineAcc, subj_list, 1000, 'pm_cost', 'pm_acc', 'mnm_mnm', 'purple')

 


################################
###### Look for outliers #######
################################

# Find outliers and remove only for specific analyses

def findOutliers(cost, measure): 
	q1 = measure.quantile(0.25)
	q3 = measure.quantile(0.75)
	iqr = q3 - q1
	return cost[~(measure < (q1 - 1.5 * iqr)) | (measure > (q3 + 1.5 * iqr)) ]

## Outliers - PM cost
#maintain_cost = findOutliers(maintain_cost, maintain_cost.pm_cost)
#monitor_cost = findOutliers(monitor_cost, monitor_cost.pm_cost)
#mnm_cost = findOutliers(mnm_cost, mnm_cost.pm_cost)

## Outliers - Accuracy for last probe
#maintain_acc = findOutliers(maintain_acc, maintain_acc.pm_acc)
#monitor_acc = findOutliers(monitor_acc, monitor_acc.pm_acc)
#mnm_acc = findOutliers(mnm_acc, mnm_acc.pm_acc)

## Outliers - RT for last probe
#maintain_pmRT = findOutliers(maintain_pmRT, maintain_pmRT.pm_probe_rt)
#monitor_pmRT = findOutliers(monitor_pmRT, monitor_pmRT.pm_probe_rt)
#mnm_pmRT = findOutliers(mnm_pmRT, mnm_pmRT.pm_probe_rt)




################################
########## Plot AICs ###########
################################
aic_cost_mnm = pd.read_csv(FIGURE_PATH+'/csvs/aic_cost_mnm.csv')

fig, aic_axes = plt.subplots(nrows=5, ncols=1,  sharex = True, sharey=True, gridspec_kw={'hspace':0})
sea.violinplot(aic_cost_mnm.main_mnm, color = 'b', ax = aic_axes[0])
sea.violinplot(aic_cost_mnm.mon_mnm, color = 'r', ax = aic_axes[1])
sea.violinplot(aic_cost_mnm.cost_mnm_noInteract, color = 'violet', ax = aic_axes[2])
sea.violinplot(aic_cost_mnm.cost_mnm_interact, color = 'mediumvioletred', ax = aic_axes[3])
sea.violinplot(aic_cost_mnm.one_mnm, color = 'g', ax = aic_axes[4])

plt.xlabel("AIC values") 
plt.savefig(FIGURE_PATH + 'modelfits.eps', dpi = 600)
plt.close()


################################
####### Remove outliers ########
################################

# Don't look at subjects where PM cost was removed for being an outlier
def removeOutliers(df): 
	for index, row in df.iterrows(): 
		if (math.isnan(row.pm_cost) or math.isnan(row.pm_acc)): 
			print(index)
			df = df.drop([index])
	return df

def removeOutliers_RT(df): 
	for index, row in df.iterrows(): 
		if (math.isnan(row.pm_cost) or math.isnan(row.pm_probe_rt)): 
			print(index)
			df = df.drop([index])
	return df

# ## Remove outliers - cost and accuracy 	

# mainCost_combineAcc = removeOutliers(mainCost_combineAcc)
# monCost_combineAcc = removeOutliers(monCost_combineAcc)
# combineCost_combineAcc = removeOutliers(combineCost_combineAcc)

# mainCost_mainAcc = removeOutliers(mainCost_mainAcc)
# monCost_monAcc = removeOutliers(monCost_monAcc)









##########################################
#Don't run below unless you care about RT#
##########################################


################################
###### Combine dataframes ######
######### cost and RT ##########
################################

# Maintain cost versus combined RT
mainCost_combineRT = pd.concat([maintain_cost, mnm_pmRT], axis = 1, sort = False)
# Monitor cost versus combined RT
monCost_combineRT = pd.concat([monitor_cost, mnm_pmRT], axis = 1, sort = False)
# Combined cost versus combined RT
combineCost_combineRT = pd.concat([mnm_cost, mnm_pmRT], axis = 1, sort = False)

# Maintain cost versus maintain RT
mainCost_mainRT = pd.concat([maintain_cost, maintain_pmRT], axis = 1, sort = False)
# Monitor cost versus monitor RT
monCost_monRT = pd.concat([monitor_cost, monitor_pmRT], axis = 1, sort = False)

# Maintain cost + Monitor cost
combined_cost = maintain_cost.pm_cost + monitor_cost.pm_cost
# Combined cost (above) versus combined RT
mplusmCost_combineRT = pd.concat([combined_cost, mnm_pmRT], axis=1, sort=False)

## Remove outliers - cost and RT

mainCost_combineRT = removeOutliers_RT(mainCost_combineRT)
monCost_combineRT = removeOutliers_RT(monCost_combineRT)
combineCost_combineRT = removeOutliers_RT(combineCost_combineRT)

mainCost_mainRT = removeOutliers_RT(mainCost_mainRT)
monCost_monRT = removeOutliers_RT(monCost_monRT)

# Drop NaNs
mainCost_combineRT = mainCost_combineRT.dropna()
monCost_combineRT = monCost_combineRT.dropna()
combineCost_combineRT = combineCost_combineRT.dropna()

mainCost_mainRT = mainCost_mainRT.dropna()
monCost_monRT = monCost_monRT.dropna()

mplusmCost_combineRT = mplusmCost_combineRT.dropna()



### Cost v RT
regPlot(combineCost_combineRT, 'Combined cost (s)','Combined PM probe RT', 'purple', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'mnmCost_v_pmRT.png', dpi = 600)
plt.close()

combine_combine_RT_lr = pg.linear_regression(combineCost_combineRT.pm_cost, combineCost_combineRT.pm_probe_rt)


### Cost v accuracy
## How does additive maintainence and monitoring cost affect performance when maintaining AND monitoring?
regPlot(mplusmCost_combineAcc, 'Maintainence + Monitoring cost (s)','Combined performance', 'purple', 'pm_acc')
plt.savefig(FIGURE_PATH + 'mplusmCost_v_pmAcc.png', dpi = 600)
plt.close()

mplusm_combineAcc_lr = pg.linear_regression(mplusmCost_combineAcc.pm_cost, mplusmCost_combineAcc.pm_acc)

### Cost v RT
regPlot(mplusmCost_combineRT, 'Maintainence + Monitoring cost (s)','Combined PM probe RT', 'purple', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'mplusmCost_v_pmRT.png', dpi = 600)
plt.close()

mplusm_combine_RT_lr = pg.linear_regression(mplusmCost_combineRT.pm_cost, mplusmCost_combineRT.pm_probe_rt)


regPlot(maintain_bootstrap_data, 'Maintain cost (s) - bootstrapped', 'Maintain performance - bootstrapped', 'b', 'pm_acc')
plt.savefig(FIGURE_PATH + 'maintain_bootstrap.png', dpi = 600)
plt.close()

maintain_bootstrap_lr = pg.linear_regression(maintain_bootstrap_data.pm_cost, maintain_bootstrap_data.pm_acc) 


regPlot(mainCost_mainAcc, 'Maintain cost (s)', 'Maintain performance', 'b', 'pm_acc')
plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainPerf.png', dpi = 600)
plt.close()

maintain_maintain_lr = pg.linear_regression(mainCost_mainAcc.pm_cost, mainCost_mainAcc.pm_acc) 

### Cost v RT
regPlot(mainCost_mainRT, 'Maintain cost (s)', 'Maintain PM probe RT', 'b', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainRT.png', dpi = 600)
plt.close()

maintain_maintain_RT_lr = pg.linear_regression(mainCost_mainRT.pm_cost, mainCost_mainRT.pm_probe_rt)




################################
########### OLD CODE ###########
################################


# maintain_cost = maintain_trials.pm_cost
# monitor_cost = monitor_trials.pm_cost
# mnm_cost = mnm_trials.pm_cost

# # returns array of [bootstrapped_mean, lower_ci, upper_ci]
# boot_maintain = bootstrapped_old(1000, maintain_cost)
# boot_monitor = bootstrapped_old(1000, monitor_cost)
# boot_mnm = bootstrapped_old(1000, mnm_cost)

# def bootstrapped_old(num_iterations, values):
# 	n_iterations = num_iterations
# 	resampled_means = []

# 	# "bootstrap" 95% confidence intervals around mean
# 	for i in range(n_iterations): 
# 		## default: np.random.choice(a, size=None, replace=True, p=None)
# 		# resample with replacement
# 		resampled_sample = np.random.choice(values, size = len(values), replace=True)
# 		resampled_mean = np.mean(resampled_sample)
# 		resampled_means.append(resampled_mean)

# 	# make sure you have the expected amount of values in list
# 	# prints 'AssertionError' if it is not true
# 	assert len(resampled_means) == n_iterations

# 	bootstrapped_mean = np.mean(resampled_means)
# 	upper_ci = np.percentile(resampled_means, 97.5)
# 	lower_ci = np.percentile(resampled_means, 2.5)

# 	return [bootstrapped_mean, lower_ci, upper_ci]
#### BOOTSTRAP

# Make a dictionary for each type of block - Maintain, Monitor, MnM
# Key = subj, value = data from all_df
# Use these dictionaries to pull individual data when looking to build new dataframes for bootstrapping

# maintain_dict = dict()
# for k, v in maintain_all.groupby('subj'):
# 	maintain_dict[k] = v

# monitor_dict = dict()
# for k, v in monitor_all.groupby('subj'):
# 	monitor_dict[k] = v

# mnm_dict = dict()
# for k, v in mnm_all.groupby('subj'):
# 	mnm_dict[k] = v

# # Array of all of the subjects
# subj_list = all_df.subj.unique() 

# n_iterations = 1000

# maintain_boot_dict = dict()

# for i in range(n_iterations):
# 	# Create a new random subject list
# 	resampled_subjList = np.random.choice(subj_list, size = len(subj_list), replace = True)

# 	# Create a dictionary for the new resampled subject list
# 	resampled_maintain_dict = dict()

# 	# For each subject in new subject list, add all of their data
# 	for subj in resampled_subjList: 
# 		resampled_maintain_dict[subj] = maintain_dict.get(subj)

# 	# Merge all of the values from the dictionary together so there is one big dataframe of all new subject data
# 	resampled_maintain_df = pd.concat(resampled_maintain_dict.values(), ignore_index = True) 

# 	maintain_bootstrap_data = resampled_maintain_df.groupby(['subj']).mean()
# 	maintain_bootstrap_lr = pg.linear_regression(maintain_bootstrap_data.pm_cost, maintain_bootstrap_data.pm_acc)
# 	maintain_boot_dict[i] = maintain_bootstrap_lr

# #
# maintain_betas = pd.DataFrame(columns = ['intercept', 'coef'])
# for key in maintain_boot_dict: 
# 	intercept = maintain_boot_dict.get(key).coef[0]
# 	coef = maintain_boot_dict.get(key).coef[1]
# 	maintain_betas.loc[key] = [intercept, coef]

