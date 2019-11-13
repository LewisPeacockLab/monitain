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

# Does maintenance cost predict combined performance? 
maintain_perform = all_df_averaged[(all_df_averaged['blockType'] == 'Maintain')]
monitor_perform = all_df_averaged[(all_df_averaged['blockType'] == 'Monitor')]
mnm_pm_perform = all_df_averaged[(all_df_averaged['blockType'] == 'MnM')]

maintain_cost = all_df_averaged[(all_df_averaged['blockType'] == 'Maintain')].groupby(['subj', 'pm_cost']).mean().reset_index()
maintain_cost = maintain_cost.drop(columns=['pm_acc', 'meanTrial_rt','og_acc'], axis=1)  

monitor_cost = all_df_averaged[(all_df_averaged['blockType'] == 'Monitor')].groupby(['subj', 'pm_cost']).mean().reset_index()
monitor_cost = monitor_cost.drop(columns=['pm_acc', 'meanTrial_rt','og_acc'], axis=1)  

mnm_cost = all_df_averaged[(all_df_averaged['blockType'] == 'MnM')].groupby(['subj', 'pm_cost']).mean().reset_index()
mnm_cost = mnm_cost.drop(columns=['pm_acc', 'meanTrial_rt','og_acc'], axis=1)

maintain_pm_perform = all_df_averaged[(all_df_averaged['blockType'] == 'Maintain')].groupby(['subj', 'pm_acc']).mean().reset_index()
maintain_pm_perform = maintain_pm_perform.drop(columns=['subj','pm_cost', 'meanTrial_rt','og_acc'], axis=1)  

monitor_pm_perform = all_df_averaged[(all_df_averaged['blockType'] == 'Monitor')].groupby(['subj', 'pm_acc']).mean().reset_index()
monitor_pm_perform = monitor_pm_perform.drop(columns=['subj','pm_cost', 'meanTrial_rt','og_acc'], axis=1)  

mnm_pm_perform = all_df_averaged[(all_df_averaged['blockType'] == 'MnM')].groupby(['subj', 'pm_acc']).mean().reset_index()
mnm_pm_perform = mnm_pm_perform.drop(columns=['subj','pm_cost', 'meanTrial_rt','og_acc'], axis=1)  

main_V_combine = pd.concat([maintain_cost, mnm_pm_perform], axis=1, sort=False)
mon_V_combine = pd.concat([monitor_cost, mnm_pm_perform], axis=1, sort=False)
mnm_V_combine = pd.concat([mnm_cost, mnm_pm_perform], axis=1, sort=False)

main_V_main = pd.concat([maintain_cost, maintain_pm_perform], axis=1, sort = False)
mon_V_mon = pd.concat([monitor_cost, monitor_pm_perform], axis = 1, sort = False)

combined_cost = maintain_cost.pm_cost + monitor_cost.pm_cost
mplusm_V_combine = pd.concat([combined_cost, mnm_pm_perform], axis=1, sort=False) 

def regPlot(combinedData, x_label, y_label, color): 
	sea.regplot(x='pm_cost', y = 'pm_acc', data = combinedData, color = color) 
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

## How does maintainance cost affect performance when only maintaining?
regPlot(main_V_main, 'Maintain cost (s)', 'Maintain performance', 'b')
plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainPerf.png', dpi = 600)
plt.close()

residPlot(main_V_main, 'Maintain cost (s)', 'Maintain performance residual', 'b')
plt.savefig(FIGURE_PATH + 'maintainCost_v_maintainPerf_residual.png', dpi = 600)
plt.close()

maintain_maintain_lr = pg.linear_regression(maintain_cost.pm_cost, maintain_pm_perform.pm_acc) 

## How does monitoring cost affect performance when only monitoring?
regPlot(mon_V_mon, 'Monitor cost (s)', 'Monitor performance', 'r')
plt.savefig(FIGURE_PATH + 'monitorCost_v_monitorPerf.png', dpi = 600)
plt.close()

residPlot(mon_V_mon, 'Monitor cost (s)', 'Monitor performance residual', 'r')
plt.savefig(FIGURE_PATH + 'monitorCost_v_monitorPerf_residual.png', dpi = 600)
plt.close()

monitor_monitor_lr = pg.linear_regression(monitor_cost.pm_cost, monitor_pm_perform.pm_acc) 


## How does maintaining cost affect PM performance?
regPlot(main_V_combine, 'Maintain cost (s)','Combined performance', 'b')
plt.savefig(FIGURE_PATH + 'maintainCost_v_pmAcc.png', dpi = 600)
plt.close()

residPlot(main_V_combine, 'Maintain cost (s)','Combined performance residual', 'b')
plt.savefig(FIGURE_PATH + 'maintainCost_v_pmAcc_residual.png', dpi = 600)
plt.close()

weight_of_maintain = pg.linear_regression(maintain_cost.pm_cost, mnm_pm_perform.pm_acc)

## How does monitoring cost affect PM performance?
regPlot(mon_V_combine, 'Monitor cost (s)','Combined performance', 'r')
plt.savefig(FIGURE_PATH + 'monitorCost_v_pmAcc.png', dpi = 600)
plt.close()

residPlot(mon_V_combine, 'Monitor cost (s)','Combined performance residual', 'r')
plt.savefig(FIGURE_PATH + 'monitorCost_v_pmAcc_residual.png', dpi = 600)
plt.close()

weight_of_monitor = pg.linear_regression(monitor_cost.pm_cost, mnm_pm_perform.pm_acc)

## How does PM cost affect performance when maintaining AND monitoring?
regPlot(mnm_V_combine, 'MnM cost (s)','Combined performance', 'purple')
plt.savefig(FIGURE_PATH + 'mnmCost_v_pmAcc.png', dpi = 600)
plt.close()

residPlot(mnm_V_combine, 'MnM cost (s)','Combined performance residual', 'purple')
plt.savefig(FIGURE_PATH + 'mnmCost_v_pmAcc_residual.png', dpi = 600)
plt.close()

weight_of_mnm = pg.linear_regression(mnm_cost.pm_cost, mnm_pm_perform.pm_acc)

## How does additive maintainence and monitoring cost affect performance when maintaining AND monitoring?
regPlot(mplusm_V_combine, 'Maintainence + Monitoring cost (s)','Combined performance', 'purple')
plt.savefig(FIGURE_PATH + 'mplusmCost_v_pmAcc.png', dpi = 600)
plt.close()

residPlot(mplusm_V_combine, 'Maintainence + Monitoring cost (s)','Combined performance residual', 'purple')
plt.savefig(FIGURE_PATH + 'mplusmCost_v_pmAcc_residual.png', dpi = 600)
plt.close()

weight_of_combined = pg.linear_regression(combined_cost, mnm_pm_perform.pm_acc)




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

