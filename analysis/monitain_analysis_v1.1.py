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


#####################################
#######  By block accuracy  #########
#####################################


###BASELINE
block1 = df_main[(df_main['block'] == 1)]
block8 = df_main[(df_main['block'] == 8)]

block1_df = pd.concat([ block1['subj'], block1['acc'], block1[rtProbes].mean(axis=1), block1[accCols].mean(axis=1)], axis=1)
block1_df['block'] = 'Baseline 1'
block1_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block8_df = pd.concat([ block8['subj'], block8['acc'], block8[rtProbes].mean(axis=1), block8[accCols].mean(axis=1)], axis=1)
block8_df['block'] = 'Baseline 2'
block8_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


# # Find mean rt for baseline blocks (CORRECT RESPONSES)
# block1_rt = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]\
# 	.groupby(['subj', 'block'])['rtProbe0'].mean()
# block8_rt = df_main[df_main['block'] == 8].groupby(['subj', 'block'])['rtProbe0'].mean()

#dfs for violin plots
#block1 = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]
#block8 = df_main[(df_main['block'] == 8) & (df_main['probe0_acc']== 1)]
baseline_df = pd.concat([block1_df, block8_df], axis = 0)#axis = 0 for horiz cat, = 1 for vert cat
baseline_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


# Compare baseline 1 to baseline 8 
ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=baseline_df, palette = "Greens", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'baseline_compare.png', dpi = 600)
plt.close()






###MAINTAIN

block2 = df_main[(df_main['block'] == 2)]
block3 = df_main[(df_main['block'] == 3)]


block2_df = pd.concat([ block2['subj'], block2['acc'], block2[rtProbes].mean(axis=1), block2[accCols].mean(axis=1)], axis=1)
block2_df['block'] = 'Maintain 1'
block2_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block3_df = pd.concat([ block3['subj'], block3['acc'], block3[rtProbes].mean(axis=1), block3[accCols].mean(axis=1)], axis=1)
block3_df['block'] = 'Maintain 2'
block3_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


maintain_df = pd.concat([block2_df, block3_df], axis=0)
maintain_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=maintain_df, palette = "Blues", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'maintain_compare_rt.png', dpi = 600)
plt.close()


# PM Accuracy
ax = sea.barplot(x='subj', y= 'pm_acc', hue= 'block', data=maintain_df, palette="Blues", ci = None)
plt.xlabel('Subject')
plt.ylabel('PM accuracy')
sea.despine()
plt.savefig(FIGURE_PATH + 'maintain_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=maintain_df, palette="Blues", ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
sea.despine()
plt.savefig(FIGURE_PATH + 'maintain_compare_ogacc.png', dpi = 600)
plt.close()





###MONITOR

block4 = df_main[(df_main['block'] == 4)]
block5 = df_main[(df_main['block'] == 5)]


block4_df = pd.concat([ block4['subj'], block4['acc'], block4[rtProbes].mean(axis=1), block4[accCols].mean(axis=1)], axis=1)
block4_df['block'] = 'Monitor 1'
block4_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block5_df = pd.concat([ block5['subj'], block5['acc'], block5[rtProbes].mean(axis=1), block5[accCols].mean(axis=1)], axis=1)
block5_df['block'] = 'Monitor 2'
block5_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


monitor_df = pd.concat([block4_df, block5_df], axis=0)
monitor_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=monitor_df, palette = "Reds", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'monitor_compare_rt.png', dpi = 600)
plt.close()

# PM Accuracy
ax = sea.barplot(x='subj', y= 'pm_acc', hue= 'block', data=monitor_df, palette="Reds", ci = None)
plt.xlabel('Subject')
plt.ylabel('PM accuracy')
sea.despine()
plt.savefig(FIGURE_PATH + 'monitor_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=monitor_df, palette="Reds", ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
sea.despine()
plt.savefig(FIGURE_PATH + 'monitor_compare_ogacc.png', dpi = 600)
plt.close()





###M&M

block6 = df_main[(df_main['block'] == 6)]
block7 = df_main[(df_main['block'] == 7)]


block6_df = pd.concat([ block6['subj'], block6['acc'], block6[rtProbes].mean(axis=1), block6[accCols].mean(axis=1)], axis=1)
block6_df['block'] = 'M&M 1'
block6_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block7_df = pd.concat([ block7['subj'], block7['acc'], block7[rtProbes].mean(axis=1), block7[accCols].mean(axis=1)], axis=1)
block7_df['block'] = 'M&M 2'
block7_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


mnm_df = pd.concat([block6_df, block7_df], axis=0)
mnm_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=mnm_df, palette = "Purples", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'mnm_compare_rt.png', dpi = 600)
plt.close()

# PM Accuracy
ax = sea.barplot(x='subj', y= 'pm_acc', hue= 'block', data=mnm_df, palette="Purples", ci = None)
plt.xlabel('Subject')
plt.ylabel('PM accuracy')
sea.despine()
plt.savefig(FIGURE_PATH + 'mnm_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=mnm_df, palette="Purples", ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
sea.despine()
plt.savefig(FIGURE_PATH + 'mnm_compare_ogacc.png', dpi = 600)
plt.close()






###ALL BLOCKS

#Get color values to set palette 
#pal = (sea.color_palette("Greens", n_colors=2))
#pal.as_hex()



my_pal = ['#aedea7', #Green
		'#abd0e6', #Blue 
		'#3787c0', #Blue 
		'#fca082', #Red 
		'#e32f27', #Red
		'#c6c7e1', #Purple
		'#796eb2', #Purple
		'#37a055'  #Green
		  ]

all_df = pd.concat([block1_df, block2_df, block3_df, block4_df, block5_df, block6_df, block7_df, block8_df], axis=0)

# for i in all_df.index: 
# 	if (all_df.loc[i, 'block'] == 1) or (all_df.loc[i,'block'] == 8): 

# 		all_df.at[i, 'block'] = str('Baseline')
# 	elif (all_df.loc[i,'block'] == 2) or (all_df.loc[i,'block'] == 3): 
# 		all_df.at[i, 'block'] = 'Maintain'
# 	elif (all_df.loc[i,'block'] == 4) or (all_df.loc[i,'block'] == 5): 
# 		all_df.at[i, 'block'] = 'Monitor'
# 	elif (all_df.loc[i,'block'] == 6) or (all_df.loc[i,'block'] == 7): 
# 		all_df.at[i, 'block'] = 'M&M'


all_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']
ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=all_df, palette = my_pal, cut = 0)
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
#legend_labels = (['Baseline 1', 'Maintain 1', 'Maintain 2',
#	'Monitor 1', 'Monitor 2', 'M&M 1', 'M&M 2', 'Baseline 2'])
#plt.legend(('Baseline 1', 'Maintain 1', 'Maintain 2',
#	'Monitor 1', 'Monitor 2', 'M&M 1', 'M&M 2', 'Baseline 2'))
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
#handles, _ = ax.get_legend_handles_labels()  
#plt.legend((),(['Baseline 1', 'Maintain 1', 'Maintain 2',
#	'Monitor 1', 'Monitor 2', 'M&M 1', 'M&M 2', 'Baseline 2'])
#plt.legend(title='Block', loc='upper center', bbox_to_anchor=(1.45, 0.8))
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_rt.png', dpi = 600)
plt.close()

# PM Accuracy
ax = sea.barplot(x='subj', y= 'pm_acc', hue= 'block', data=all_df, palette=my_pal, ci = None)
plt.xlabel('Subject')
plt.ylabel('PM accuracy')
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=all_df, palette=my_pal, ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_ogacc.png', dpi = 600)
plt.close() 
 







#allTogether_df = pd.concat([block1_df, block2_df, block3_df, block4_df, block5_df, block6_df, block7_df, block8_df], axis=0)


#allTogether_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'block']
ax = sea.violinplot(x='block', y = 'meanTrial_rt', data=all_df, palette = my_pal, cut = 0)
plt.xlabel('Block')
plt.ylabel('Reaction time (s)')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_rt.png', dpi = 600)
plt.close()

# PM Accuracy
ax = sea.barplot(x='block', y= 'pm_acc', data=all_df, palette=my_pal, ci = None)
plt.xlabel('Block')
plt.ylabel('PM accuracy')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_pmacc.png', dpi = 600)
plt.close() 

all_df = pd.concat([block1_df, block2_df, block3_df, block4_df, block5_df, block6_df, block7_df, block8_df], axis=0)

# for i in all_df.index: 
# 	if (all_df.loc[i, 'block'] == 1) or (all_df.loc[i,'block'] == 8): 

# 		all_df.at[i, 'block'] = str('Baseline')
# 	elif (all_df.loc[i,'block'] == 2) or (all_df.loc[i,'block'] == 3): 
# 		all_df.at[i, 'block'] = 'Maintain'
# 	elif (all_df.loc[i,'block'] == 4) or (all_df.loc[i,'block'] == 5): 
# 		all_df.at[i, 'block'] = 'Monitor'
# 	elif (all_df.loc[i,'block'] == 6) or (all_df.loc[i,'block'] == 7): 
# 		all_df.at[i, 'block'] = 'M&M'


all_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']
ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=all_df, palette = my_pal, cut = 0)
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
#legend_labels = (['Baseline 1', 'Maintain 1', 'Maintain 2',
#	'Monitor 1', 'Monitor 2', 'M&M 1', 'M&M 2', 'Baseline 2'])
#plt.legend(('Baseline 1', 'Maintain 1', 'Maintain 2',
#	'Monitor 1', 'Monitor 2', 'M&M 1', 'M&M 2', 'Baseline 2'))
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
#handles, _ = ax.get_legend_handles_labels()  
#plt.legend((),(['Baseline 1', 'Maintain 1', 'Maintain 2',
#	'Monitor 1', 'Monitor 2', 'M&M 1', 'M&M 2', 'Baseline 2'])
#plt.legend(title='Block', loc='upper center', bbox_to_anchor=(1.45, 0.8))
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_rt.png', dpi = 600)
plt.close()

all_df_minusBase = all_df.copy()
all_df_minusBase.drop(all_df_minusBase[all_df_minusBase['block'] == "Baseline 2"].index, inplace=True)
all_df_minusBase.drop(all_df_minusBase[all_df_minusBase['block'] == "Baseline 1"].index, inplace=True)


# PM Accuracy
ax = sea.barplot(x='subj', y= 'pm_acc', hue= 'block', data=all_df_minusBase, palette=my_pal[1:-1], ci = None)
plt.xlabel('Subject')
plt.ylabel('PM accuracy')
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=all_df, palette=my_pal, ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_ogacc.png', dpi = 600)
plt.close() 
 


#allTogether_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'block']
ax = sea.violinplot(x='block', y = 'meanTrial_rt', data=all_df, palette = my_pal, cut = 0)
plt.xlabel('Block')
plt.ylabel('Reaction time (s)')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_rt.png', dpi = 600)
plt.close()

# PM Accuracy
ax = sea.barplot(x='block', y= 'pm_acc', data=all_df_minusBase, palette=my_pal[1:-1], ci = None)
plt.xlabel('Block')
plt.ylabel('PM accuracy')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='block', y= 'og_acc', data=all_df, palette=my_pal, ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_ogacc.png', dpi = 600)
plt.close() 



#### PM cost

base_cost = (baseline_df.groupby('subj')['meanTrial_rt'].mean())

maintain1_cost = block2_df.groupby('subj')['meanTrial_rt'].mean()
maintain2_cost = block3_df.groupby('subj')['meanTrial_rt'].mean()

maintain1_cost_PM = maintain1_cost - base_cost
maintain1_cost_PM = maintain1_cost_PM.to_frame().reset_index()
maintain1_cost_PM['block'] = 'Maintain 1'

maintain2_cost_PM = maintain2_cost - base_cost
maintain2_cost_PM = maintain2_cost_PM.to_frame().reset_index()
maintain2_cost_PM['block'] = 'Maintain 2'

monitor1_cost = block4_df.groupby('subj')['meanTrial_rt'].mean()
monitor2_cost = block5_df.groupby('subj')['meanTrial_rt'].mean()

monitor1_cost_PM = monitor1_cost - base_cost
monitor1_cost_PM = monitor1_cost_PM.to_frame().reset_index()
monitor1_cost_PM['block'] = 'Monitor 1'

monitor2_cost_PM = monitor2_cost - base_cost
monitor2_cost_PM = monitor2_cost_PM.to_frame().reset_index()
monitor2_cost_PM['block'] = 'Monitor 2'

mnm1_cost = block6_df.groupby('subj')['meanTrial_rt'].mean()
mnm2_cost = block7_df.groupby('subj')['meanTrial_rt'].mean()

mnm1_cost_PM = mnm1_cost - base_cost
mnm1_cost_PM = mnm1_cost_PM.to_frame().reset_index()
mnm1_cost_PM['block'] = 'M&M 1'

mnm2_cost_PM = mnm2_cost - base_cost
mnm2_cost_PM = mnm2_cost_PM.to_frame().reset_index()
mnm2_cost_PM['block'] = 'M&M 2'

pmCost = pd.concat([maintain1_cost_PM, maintain2_cost_PM, 
		  monitor1_cost_PM, monitor2_cost_PM, 
		  mnm1_cost_PM, mnm2_cost_PM], axis = 0)

# PM cost
ax = sea.barplot(x='block', y= 'meanTrial_rt', data=pmCost, palette=my_pal[1:-1], ci = None)
plt.xlabel('Block')
plt.ylabel('PM cost (s)')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_PMCOST.png', dpi = 600)
plt.close()

#####################################
############  Figures  ##############
#####################################








