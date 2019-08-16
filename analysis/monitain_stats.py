import os
import pingouin as pg
import pandas as pd
import numpy as np

### run repeated measure ANOVAs and post hoc t-tests
PATH = os.path.expanduser('~')
FIGURE_PATH = PATH + '/monitain/analysis/output/'
CSV_PATH = FIGURE_PATH + 'csvs'

all_df_averaged = pd.read_csv(FIGURE_PATH+'/csvs/ALL.csv')
pmCost_df_averaged = pd.read_csv(FIGURE_PATH+'/csvs/PM_COST.csv')
all_df_averaged_minusBase = all_df_averaged[all_df_averaged.blockType != 'Baseline']   

## og acc
aov_og = pg.rm_anova(dv='og_acc', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_og = pg.pairwise_ttests(dv='og_acc', within='blockType', data=all_df_averaged)

## pm acc
aov_pm = pg.rm_anova(dv='pm_acc', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_pm = pg.pairwise_ttests(dv='pm_acc', within='blockType', subject = 'subj', data=all_df_averaged)

## rt
aov_rt = pg.rm_anova(dv='meanTrial_rt', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_rt = pg.pairwise_ttests(dv='meanTrial_rt', within='blockType', subject = 'subj', data=all_df_averaged)

## pm cost
aov_pmCost = pg.rm_anova(dv='pm_cost', within = 'blockType', subject = 'subj', data=pmCost_df_averaged, detailed = True)
posthoc_pmCost = pg.pairwise_ttests(dv='pm_cost', within='blockType', subject = 'subj', data=pmCost_df_averaged)


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
		p_val_list_ttests.append(posthoc_type['p-unc'][i])
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
ttest_data = {'p-unc': p_val_list_ttests, 'condition': condition_list_ttests, 'sig code': sig_list_ttests}
ttest_df = pd.DataFrame(ttest_data, index = data_indx_ttest)
# export to csv
fname_ttest = os.path.join('ttest_pvals.csv')
ttest_df.to_csv(fname_ttest)


### LOGISTIC REGRESSION
X = all_df_averaged
lom = logistic_regression(X = )

