##################################
##################################
######## monitain STATS ##########
########### for v.1.1 ############
######## Katie Hedgpeth ##########
############ updated #############
######### November 2019 ##########
##################################

### This script will: 
# run repeated measure ANOVAs and post hoc t-tests 

import os
import pingouin as pg
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
import math

#import bootstrapped.bootstrap as baseline
#import bootstrapped.stats_functions as bs_stats

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

X = all_df_byTrial.pm_cost
y = all_df_byTrial.pm_acc
pg.logistic_regression(X, y, remove_na=True)  

## create scatter plot of x = pm cost and y = pm acc
# ax = sea.barplot(x="pm_acc", y = "pm_cost", data=all_df_byTrial) 
# plt.xlabel('PM accuracy')
# plt.ylabel('PM cost (secs)')
# plt.savefig(FIGURE_PATH + 'pmAcc_v_pmCost.png', dpi = 600)
# plt.close()

############

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








# Array of all of the subjects
subj_list = all_df_byTrial.subj.unique() 



def bootstrapped(trial_df, subj_list, n_iterations, x, y, type):

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
		resampled_df = pd.concat(resampled_dict.values(), ignore_index = False) 

		bootstrap_data = resampled_df.groupby(['subj']).mean()
		bootstrap_lr = pg.linear_regression(bootstrap_data[x], bootstrap_data[y])
		boot_dict[i] = bootstrap_lr

	# Create df of linear regression results - include intercept and coefficient for each bootstrap iteration
	betas = pd.DataFrame(columns = ['intercept', 'coef'])
	for key in boot_dict: 
		intercept = boot_dict.get(key).coef[0]
		coef = boot_dict.get(key).coef[1]
		betas.loc[key] = [intercept, coef]


	# Plot betas	
	sea.distplot(betas.coef) 
	plt.xlabel('Coefficient')
	plt.savefig(FIGURE_PATH + type + '_bootstrap.png', dpi = 600)
	plt.close()

	return betas;



maintain_maintain_betas = bootstrapped(maintainCost_maintainAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_maintain')
monitor_monitor_betas = bootstrapped(monitorCost_monitorAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_monitor')
mnm_mnm_betas = bootstrapped(mnmCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'mnm_mnm')

maintain_mnm_betas = bootstrapped(maintainCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_mnm')
monitor_mnm_betas = bootstrapped(monitorCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_mnm')








##################


# Does maintenance cost predict combined performance? 
maintain_trials = all_df_averaged[(all_df_averaged['blockType'] == 'Maintain')] # maintain block trials
monitor_trials = all_df_averaged[(all_df_averaged['blockType'] == 'Monitor')] # monitor block trials
mnm_trials = all_df_averaged[(all_df_averaged['blockType'] == 'MnM')] # mnm block trials

maintain_results = maintain_trials.groupby(['subj']).mean().reset_index()
maintain_cost = maintain_results.drop(columns=['pm_acc', 'meanTrial_rt','og_acc', 'pm_probe_rt'], axis=1)
maintain_acc = maintain_results.drop(columns=['subj', 'pm_cost', 'meanTrial_rt','og_acc', 'pm_probe_rt'], axis=1) 
maintain_pmRT = maintain_results.drop(columns=['subj', 'pm_cost', 'meanTrial_rt', 'og_acc', 'pm_acc'])
# Remove 'subj' column if you need a subj num indication in accuracy df

monitor_results = monitor_trials.groupby(['subj']).mean().reset_index()
monitor_cost = monitor_results.drop(columns=['pm_acc', 'meanTrial_rt','og_acc', 'pm_probe_rt'], axis=1)
monitor_acc = monitor_results.drop(columns=['subj', 'pm_cost', 'meanTrial_rt','og_acc', 'pm_probe_rt'], axis=1) 
monitor_pmRT = monitor_results.drop(columns=['subj', 'pm_cost', 'meanTrial_rt', 'og_acc', 'pm_acc'])

mnm_results = mnm_trials.groupby(['subj']).mean().reset_index()
mnm_cost = mnm_results.drop(columns=['pm_acc', 'meanTrial_rt','og_acc', 'pm_probe_rt'], axis=1)
mnm_acc = mnm_results.drop(columns=['subj', 'pm_cost', 'meanTrial_rt','og_acc', 'pm_probe_rt'], axis=1) 
mnm_pmRT = maintain_results.drop(columns=['subj', 'pm_cost', 'meanTrial_rt', 'og_acc', 'pm_acc'])


##############

### LOOKING FOR OUTLIERS

# Find outliers and remove only for specific analyses

def findOutliers(cost, measure): 
	q1 = measure.quantile(0.25)
	q3 = measure.quantile(0.75)
	iqr = q3 - q1
	return cost[~(measure < (q1 - 1.5 * iqr)) | (measure > (q3 + 1.5 * iqr)) ]

## Outliers - PM cost
maintain_cost = findOutliers(maintain_cost, maintain_cost.pm_cost)
monitor_cost = findOutliers(monitor_cost, monitor_cost.pm_cost)
mnm_cost = findOutliers(mnm_cost, mnm_cost.pm_cost)

## Outliers - Accuracy for last probe
maintain_acc = findOutliers(maintain_acc, maintain_acc.pm_acc)
monitor_acc = findOutliers(monitor_acc, monitor_acc.pm_acc)
mnm_acc = findOutliers(mnm_acc, mnm_acc.pm_acc)

## Outliers - RT for last probe
maintain_pmRT = findOutliers(maintain_pmRT, maintain_pmRT.pm_probe_rt)
monitor_pmRT = findOutliers(monitor_pmRT, monitor_pmRT.pm_probe_rt)
mnm_pmRT = findOutliers(mnm_pmRT, mnm_pmRT.pm_probe_rt)


## Combine DFs - cost and accuracy

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

## Combine DFs - cost and RT

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

## Remove outliers - cost and accuracy 	

mainCost_combineAcc = removeOutliers(mainCost_combineAcc)
monCost_combineAcc = removeOutliers(monCost_combineAcc)
combineCost_combineAcc = removeOutliers(combineCost_combineAcc)

mainCost_mainAcc = removeOutliers(mainCost_mainAcc)
monCost_monAcc = removeOutliers(monCost_monAcc)

# Drop NaNs
mainCost_combineAcc = mainCost_combineAcc.dropna()
monCost_combineAcc = monCost_combineAcc.dropna()
combineCost_combineAcc = combineCost_combineAcc.dropna()

mainCost_mainAcc = mainCost_mainAcc.dropna()
monCost_monAcc = monCost_monAcc.dropna()

mplusmCost_combineAcc = mplusmCost_combineAcc.dropna()


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



def regPlot(combinedData, x_label, y_label, color, y_data): 
	sea.regplot(x='pm_cost', y = y_data, data = combinedData, color = color) 
	plt.xlabel(x_label)
	plt.ylabel(y_label)

def residPlot(combinedData, x_label, y_label, color):
	sea.residplot(x = 'pm_cost', y = 'pm_acc', data = main_V_main, color = color, scatter_kws = {"s": 80}) 
	plt.xlabel(x_label)
	plt.ylabel(y_label)

##TODO: Change file extension based on what the image is for
# png for Slack
# pdf for viewing
# eps for editing


### Cost v accuracy
## How does maintainance cost affect performance when only maintaining?
regPlot(mainCost_mainAcc, 'Maintain cost (s)', 'Maintain performance', 'b', 'pm_acc')
plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainPerf.png', dpi = 600)
plt.close()

maintain_maintain_lr = pg.linear_regression(mainCost_mainAcc.pm_cost, mainCost_mainAcc.pm_acc) 

### Cost v RT
regPlot(mainCost_mainRT, 'Maintain cost (s)', 'Maintain PM probe RT', 'b', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainRT.png', dpi = 600)
plt.close()

maintain_maintain_RT_lr = pg.linear_regression(mainCost_mainRT.pm_cost, mainCost_mainRT.pm_probe_rt)


### Cost v accuracy
## How does monitoring cost affect performance when only monitoring?
regPlot(monCost_monAcc, 'Monitor cost (s)', 'Monitor performance', 'r', 'pm_acc')
plt.savefig(FIGURE_PATH + 'monitorCost_v_monitorPerf.png', dpi = 600)
plt.close()

monitor_monitor_lr = pg.linear_regression(monCost_monAcc.pm_cost, monCost_monAcc.pm_acc)

### Cost v RT
regPlot(monCost_monRT, 'Monitor cost (s)', 'Monitor PM probe RT', 'r', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'monitorCost_v_monitorRT.png', dpi = 600)
plt.close()

monitor_monitor_RT_lr = pg.linear_regression(monCost_monRT.pm_cost, monCost_monRT.pm_probe_rt)



### Cost v accuracy
## How does maintaining cost affect PM performance?
regPlot(mainCost_combineAcc, 'Maintain cost (s)','Combined performance', 'b', 'pm_acc')
plt.savefig(FIGURE_PATH + 'maintainCost_v_pmAcc.png', dpi = 600)
plt.close()

maintain_combine_lr = pg.linear_regression(mainCost_combineAcc.pm_cost, mainCost_combineAcc.pm_acc)

### Cost v RT
regPlot(mainCost_combineRT, 'Maintain cost (s)','Combined PM probe RT', 'b', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'maintainCost_v_pmRT.png', dpi = 600)
plt.close()

maintain_combine_RT_lr = pg.linear_regression(mainCost_combineRT.pm_cost, mainCost_combineRT.pm_probe_rt)


### Cost v accuracy
## How does monitoring cost affect PM performance?
regPlot(monCost_combineAcc, 'Monitor cost (s)','Combined performance', 'r', 'pm_acc')
plt.savefig(FIGURE_PATH + 'monitorCost_v_pmAcc.png', dpi = 600)
plt.close()

monitor_combine_lr = pg.linear_regression(monCost_combineAcc.pm_cost, monCost_combineAcc.pm_acc)

### Cost v RT
regPlot(monCost_combineRT, 'Monitor cost (s)','Combined PM probe RT', 'r', 'pm_probe_rt')
plt.savefig(FIGURE_PATH + 'monitorCost_v_pmRT.png', dpi = 600)
plt.close()

monitor_combine_RT_lr = pg.linear_regression(monCost_combineRT.pm_cost, monCost_combineRT.pm_probe_rt)


### Cost v accuracy
## How does PM cost affect performance when maintaining AND monitoring?
regPlot(combineCost_combineAcc, 'MnM cost (s)','Combined performance', 'purple', 'pm_acc')
plt.savefig(FIGURE_PATH + 'mnmCost_v_pmAcc.png', dpi = 600)
plt.close()

combine_combine_lr = pg.linear_regression(combineCost_combineAcc.pm_cost, combineCost_combineAcc.pm_acc)

### Cost v RT
regPlot(combineCost_combineRT, 'MnM cost (s)','Combined PM probe RT', 'purple', 'pm_probe_rt')
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


#### BOOTSTRAP

# Make a dictionary for each type of block - Maintain, Monitor, MnM
# Key = subj, value = data from all_df
# Use these dictionaries to pull individual data when looking to build new dataframes for bootstrapping

maintain_dict = dict()
for k, v in maintain_all.groupby('subj'):
	maintain_dict[k] = v

monitor_dict = dict()
for k, v in monitor_all.groupby('subj'):
	monitor_dict[k] = v

mnm_dict = dict()
for k, v in mnm_all.groupby('subj'):
	mnm_dict[k] = v

# Array of all of the subjects
subj_list = all_df.subj.unique() 

n_iterations = 1000

maintain_boot_dict = dict()

for i in range(n_iterations):
	# Create a new random subject list
	resampled_subjList = np.random.choice(subj_list, size = len(subj_list), replace = True)

	# Create a dictionary for the new resampled subject list
	resampled_maintain_dict = dict()

	# For each subject in new subject list, add all of their data
	for subj in resampled_subjList: 
		resampled_maintain_dict[subj] = maintain_dict.get(subj)

	# Merge all of the values from the dictionary together so there is one big dataframe of all new subject data
	resampled_maintain_df = pd.concat(resampled_maintain_dict.values(), ignore_index = True) 

	maintain_bootstrap_data = resampled_maintain_df.groupby(['subj']).mean()
	maintain_bootstrap_lr = pg.linear_regression(maintain_bootstrap_data.pm_cost, maintain_bootstrap_data.pm_acc)
	maintain_boot_dict[i] = maintain_bootstrap_lr

#
maintain_betas = pd.DataFrame(columns = ['intercept', 'coef'])
for key in maintain_boot_dict: 
	intercept = maintain_boot_dict.get(key).coef[0]
	coef = maintain_boot_dict.get(key).coef[1]
	maintain_betas.loc[key] = [intercept, coef]

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







maintain_cost = maintain_trials.pm_cost
monitor_cost = monitor_trials.pm_cost
mnm_cost = mnm_trials.pm_cost

# returns array of [bootstrapped_mean, lower_ci, upper_ci]
boot_maintain = bootstrapped_old(1000, maintain_cost)
boot_monitor = bootstrapped_old(1000, monitor_cost)
boot_mnm = bootstrapped_old(1000, mnm_cost)

def bootstrapped_old(num_iterations, values):
	n_iterations = num_iterations
	resampled_means = []

	# "bootstrap" 95% confidence intervals around mean
	for i in range(n_iterations): 
		## default: np.random.choice(a, size=None, replace=True, p=None)
		# resample with replacement
		resampled_sample = np.random.choice(values, size = len(values), replace=True)
		resampled_mean = np.mean(resampled_sample)
		resampled_means.append(resampled_mean)

	# make sure you have the expected amount of values in list
	# prints 'AssertionError' if it is not true
	assert len(resampled_means) == n_iterations

	bootstrapped_mean = np.mean(resampled_means)
	upper_ci = np.percentile(resampled_means, 97.5)
	lower_ci = np.percentile(resampled_means, 2.5)

	return [bootstrapped_mean, lower_ci, upper_ci]


#### BY TRIAL 

maintain_perform_byTrial = all_df_byTrial[(all_df_byTrial['blockType'] == 'Maintain')]
monitor_perform_byTrial = all_df_byTrial[(all_df_byTrial['blockType'] == 'Monitor')]
mnm_pm_perform_byTrial = all_df_byTrial[(all_df_byTrial['blockType'] == 'MnM')]


## not sure what these are for
##maintain_array = maintain_cost.array
##monitor_array = monitor_cost.array
##mnm_pm_array = mnm_pm_perform.array


## not sure about this either
##ax = sea.pointplot(x=all_df_averaged[(all_df_averaged['blockType'].pm_cost == 'Maintain')], y = all_df_averaged[(all_df_averaged['blockType'].pm_acc == 'MnM')], data = all_df_averaged) 







# Does monitoring cost predict combined performance? 

