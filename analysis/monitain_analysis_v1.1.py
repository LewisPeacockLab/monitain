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
data_dir = PATH+'/Dropbox (LewPeaLab)/BEHAVIOR/monitain/'
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

# Change block values so v1.0 has block type attached 
df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 1), 
	'block'] = "('base1', 1)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 2), 
	'block'] = "('maintain1', 2)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 3), 
	'block'] = "('maintain2', 3)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 4), 
	'block'] = "('monitor1', 4)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 5), 
	'block'] = "('monitor2', 5)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 6), 
	'block'] = "('mnm1', 6)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 7), 
	'block'] = "('mnm2', 7)"

df_main.loc[ (df_main.version == 'vers1.0')
	& (df_main.block == 8), 
	'block'] = "('base2', 8)"


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
block_base1 = df_main[(df_main['block'] == "('base1', 1)")]
block_base2 = df_main[(df_main['block'] == "('base2', 8)")]

block_base1_df = pd.concat([ block_base1['subj'], block_base1['version'], block_base1['acc'], block_base1[rtProbes].mean(axis=1), block_base1[accCols].mean(axis=1)], axis=1)
block_base1_df['block'] = 'Baseline 1'
block_base1_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block_base2_df = pd.concat([ block_base2['subj'], block_base2['version'], block_base2['acc'], block_base2[rtProbes].mean(axis=1), block_base2[accCols].mean(axis=1)], axis=1)
block_base2_df['block'] = 'Baseline 2'
block_base2_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


# # Find mean rt for baseline blocks (CORRECT RESPONSES)
# block_base1_rt = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]\
# 	.groupby(['subj', 'block'])['rtProbe0'].mean()
# block_base2_rt = df_main[df_main['block'] == 8].groupby(['subj', 'block'])['rtProbe0'].mean()

#dfs for violin plots
#block_base1 = df_main[(df_main['block'] == 1) & (df_main['probe0_acc']== 1)]
#block_base2 = df_main[(df_main['block'] == 8) & (df_main['probe0_acc']== 1)]
baseline_df = pd.concat([block_base1_df, block_base2_df], axis = 0)#axis = 0 for horiz cat, = 1 for vert cat
baseline_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


# Compare baseline 1 to baseline 8 
ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=baseline_df, palette = "Greens", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'baseline_compare.png', dpi = 600)
plt.close()






###MAINTAIN

block_maintain1 = df_main[(df_main['block'] == "('maintain1', 2)")]
block_maintain2 = df_main[(df_main['block'] == "('maintain2', 3)") | (df_main['block'] == "('maintain2', 5)")]


block_maintain1_df = pd.concat([ block_maintain1['subj'], block_maintain1['version'], block_maintain1['acc'], block_maintain1[rtProbes].mean(axis=1), block_maintain1[accCols].mean(axis=1)], axis=1)
block_maintain1_df['block'] = 'Maintain 1'
block_maintain1_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block_maintain2_df = pd.concat([ block_maintain2['subj'], block_maintain2['version'], block_maintain2['acc'], block_maintain2[rtProbes].mean(axis=1), block_maintain2[accCols].mean(axis=1)], axis=1)
block_maintain2_df['block'] = 'Maintain 2'
block_maintain2_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


maintain_df = pd.concat([block_maintain1_df, block_maintain2_df], axis=0)
maintain_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

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

block_monitor1 = df_main[(df_main['block'] == "('monitor1', 4)") | (df_main['block'] == "('monitor1', 3)")]
block_monitor2 = df_main[(df_main['block'] == "('monitor2', 5)") | (df_main['block'] == "('monitor2', 6)")]


block_monitor1_df = pd.concat([ block_monitor1['subj'], block_monitor1['version'], block_monitor1['acc'], block_monitor1[rtProbes].mean(axis=1), block_monitor1[accCols].mean(axis=1)], axis=1)
block_monitor1_df['block'] = 'Monitor 1'
block_monitor1_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block_monitor2_df = pd.concat([ block_monitor2['subj'], block_monitor2['version'], block_monitor2['acc'], block_monitor2[rtProbes].mean(axis=1), block_monitor2[accCols].mean(axis=1)], axis=1)
block_monitor2_df['block'] = 'Monitor 2'
block_monitor2_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


monitor_df = pd.concat([block_monitor1_df, block_monitor2_df], axis=0)
monitor_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

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

block_mnm1 = df_main[(df_main['block'] == "('mnm1', 6)") | (df_main['block'] == "('mnm1', 4)")]
block_mnm2 = df_main[(df_main['block'] == "('mnm2', 7)")] 


block_mnm1_df = pd.concat([ block_mnm1['subj'], block_mnm1['version'], block_mnm1['acc'], block_mnm1[rtProbes].mean(axis=1), block_mnm1[accCols].mean(axis=1)], axis=1)
block_mnm1_df['block'] = 'M&M 1'
block_mnm1_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

block_mnm2_df = pd.concat([ block_mnm2['subj'], block_mnm2['version'], block_mnm2['acc'], block_mnm2[rtProbes].mean(axis=1), block_mnm2[accCols].mean(axis=1)], axis=1)
block_mnm2_df['block'] = 'M&M 2'
block_mnm2_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']


mnm_df = pd.concat([block_mnm1_df, block_mnm2_df], axis=0)
mnm_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=mnm_df, palette = "Purples", cut = 0)
# Cut = 0 so range is limited to observed data
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig(FIGURE_PATH + 'mnm_compare_rt.png', dpi = 600)
plt.close()

#test_xx = mnm_df.subj == 'sxx'
#mnm_df[test_xx].mean()

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

my_pal_paired = ['#aedea7', #Green
		'#7bc96f', #Darker green
		'#abd0e6', #Blue 
		'#70afd4', #Darker blue
		'#3787c0', #Blue 
		'##265d85', #Darker blue
		'#fca082', #Red, 
		'#fa6737', #Darker red 
		'#e32f27', #Red
		'#a81c16', #Darker red
		'#c6c7e1', #Purple
		'#9496c7', #Darker purple
		'#796eb2', #Purple
		'#54498a', #Darker purple
		'#37a055', #Green
		'#236737' #Darker green
		  ]

block_base1_df_v10 = block_base1_df[block_base1_df['version'] == 'vers1.0']
block_base1_df_v11 = block_base1_df[block_base1_df['version'] == 'vers1.1']

block_maintain1_df_v10 = block_maintain1_df[block_maintain1_df['version'] == 'vers1.0']
block_maintain1_df_v11 = block_maintain1_df[block_maintain1_df['version'] == 'vers1.1']

block_maintain2_df_v10 = block_maintain2_df[block_maintain2_df['version'] == 'vers1.0']
block_maintain2_df_v11 = block_maintain2_df[block_maintain2_df['version'] == 'vers1.1']

block_monitor1_df_v10 = block_monitor1_df[block_monitor1_df['version'] == 'vers1.0']
block_monitor1_df_v11 = block_monitor1_df[block_monitor1_df['version'] == 'vers1.1']

block_monitor2_df_v10 = block_monitor2_df[block_monitor2_df['version'] == 'vers1.0']
block_monitor2_df_v11 = block_monitor2_df[block_monitor2_df['version'] == 'vers1.1']

block_mnm1_df_v10 = block_mnm1_df[block_mnm1_df['version'] == 'vers1.0']
block_mnm1_df_v11 = block_mnm1_df[block_mnm1_df['version'] == 'vers1.1']

block_mnm2_df_v10 = block_mnm2_df[block_mnm2_df['version'] == 'vers1.0']
block_mnm2_df_v11 = block_mnm2_df[block_mnm2_df['version'] == 'vers1.1']

block_base2_df_v10 = block_base2_df[block_base2_df['version'] == 'vers1.0']
block_base2_df_v11 = block_base2_df[block_base2_df['version'] == 'vers1.1']






all_df = pd.concat([block_base1_df, block_maintain1_df, block_maintain2_df, block_monitor1_df, block_monitor2_df, block_mnm1_df, block_mnm2_df, block_base2_df], axis=0)

all_df_v10 = pd.concat([block_base1_df_v10, block_maintain1_df_v10, block_maintain2_df_v10, block_monitor1_df_v10, block_monitor2_df_v10, block_mnm1_df_v10, block_mnm2_df_v10, block_base2_df_v10], axis=0)
all_df_v11 = pd.concat([block_base1_df_v11, block_maintain1_df_v11, block_maintain2_df_v11, block_monitor1_df_v11, block_monitor2_df_v11, block_mnm1_df_v11, block_mnm2_df_v11, block_base2_df_v11], axis=0)



all_df_combined = pd.concat([all_df_v10, all_df_v11], axis = 0)
# for i in all_df.index: 
# 	if (all_df.loc[i, 'block'] == 1) or (all_df.loc[i,'block'] == 8): 

# 		all_df.at[i, 'block'] = str('Baseline')
# 	elif (all_df.loc[i,'block'] == 2) or (all_df.loc[i,'block'] == 3): 
# 		all_df.at[i, 'block'] = 'Maintain'
# 	elif (all_df.loc[i,'block'] == 4) or (all_df.loc[i,'block'] == 5): 
# 		all_df.at[i, 'block'] = 'Monitor'
# 	elif (all_df.loc[i,'block'] == 6) or (all_df.loc[i,'block'] == 7): 
# 		all_df.at[i, 'block'] = 'M&M'


all_df.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']

all_df_combined.columns = all_df.columns


#######################
## Analysis to try: 
#all_df.groupby(['subj', 'pm_acc']).mean()
## groups by subj and pm acc so you can so what everything else looks like by subject 
#when they are correct and when the aren't correct on the pm task

##ADD BACK LATER IF NOT EXCLUDING ANYONE:
# grouped = all_df.groupby('subj')['pm_acc'].mean()

# removeSubjs = []
# for subj, pmAcc in grouped.iteritems(): 
# 	if pmAcc < 0.5: 
# 		print subj #this tells us who to remove
# 		removeSubjs.append(subj)
# 		print removeSubjs

# for i in range(len(removeSubjs)):
# 	#grouped = grouped.drop([removeSubjs[i]])
# 	all_df_combined = all_df_combined[all_df_combined.subj != removeSubjs[i]]
# 	all_df = all_df[all_df.subj != removeSubjs[i]]



#######################

#Replace RT with nan in resp was quicker than 0.05 secs, also replace pm acc with nan
#Cut off based on findings from "Eye Movements and Visual Encoding during Scene Perception"
#Rayner et al., 2009
for trial in all_df_combined.index: 
	if all_df_combined.loc[trial, 'meanTrial_rt'] < 0.05: 
		print all_df_combined.loc[trial, 'meanTrial_rt']
		all_df_combined.at[trial, 'meanTrial_rt'] = np.nan
		all_df_combined.at[trial, 'pm_acc'] = np.nan

ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=all_df_combined, palette = my_pal, cut = 0)
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
 




all_df_v10 = pd.concat([block_base1_df, block_maintain1_df, block_maintain2_df, block_monitor1_df, block_monitor2_df, block_mnm1_df, block_mnm2_df, block_base2_df], axis=0)
all_df_v11 = pd.concat([block_base1_df, block_maintain1_df, block_maintain2_df, block_monitor1_df, block_monitor2_df, block_mnm1_df, block_mnm2_df, block_base2_df], axis=0)




#allTogether_df = pd.concat([block_base1_df, block_maintain1_df, block_maintain2_df, block_monitor1_df, block_monitor2_df, block_mnm1_df, block_mnm2_df, block_base2_df], axis=0)

for i in all_df_combined.index: 
	if (all_df_combined.loc[i, 'block'] == 'Baseline 1') or (all_df_combined.loc[i,'block'] == 'Baseline 2'): 
		all_df_combined.at[i, 'block'] = 'Baseline'
	elif (all_df_combined.loc[i,'block'] == 'Maintain 1') or (all_df_combined.loc[i,'block'] == 'Maintain 2'): 
		all_df_combined.at[i, 'block'] = 'Maintain'
	elif (all_df_combined.loc[i,'block'] == 'Monitor 1') or (all_df_combined.loc[i,'block'] == 'Monitor 2'): 
		all_df_combined.at[i, 'block'] = 'Monitor'
	elif (all_df_combined.loc[i,'block'] == 'M&M 1') or (all_df_combined.loc[i,'block'] == 'M&M 2'): 
		all_df_combined.at[i, 'block'] = 'M&M'


#allTogether_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'block']
ax = sea.violinplot(x='block', y = 'meanTrial_rt', hue = 'version', data=all_df_combined, palette = my_pal, cut = 0, split = True)
plt.xlabel('Block')
plt.ylabel('Reaction time (s)')
ax.legend(title = 'Version')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_rt.png', dpi = 600)
plt.close()



# PM Accuracy
ax = sea.barplot(x='block', y= 'pm_acc', hue = 'version', data=all_df_combined, palette=my_pal, ci = None)
plt.xlabel('Block')
plt.ylabel('PM accuracy')
ax.legend(title = 'Version')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_pmacc.png', dpi = 600)
plt.close() 





all_df_combined.columns = ['subj', 'version', 'pm_acc', 'meanTrial_rt', 'og_acc', 'block']
ax = sea.violinplot(x='subj', y = 'meanTrial_rt', hue = 'block', data=all_df_combined, palette = my_pal, cut = 0)
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

#all_df_minusBase = all_df.copy()
#all_df_minusBase.drop(all_df_minusBase[all_df_minusBase['block'] == "Baseline 2"].index, inplace=True)
#all_df_minusBase.drop(all_df_minusBase[all_df_minusBase['block'] == "Baseline 1"].index, inplace=True)

all_df_combined_minusBase = all_df_combined.copy()
all_df_combined_minusBase.drop(all_df_combined_minusBase[all_df_combined_minusBase['block'] == "Baseline 2"].index, inplace=True)
all_df_combined_minusBase.drop(all_df_combined_minusBase[all_df_combined_minusBase['block'] == "Baseline 1"].index, inplace=True)

# PM Accuracy
ax = sea.barplot(x='subj', y= 'pm_acc', hue= 'block', data=all_df_combined_minusBase, palette=my_pal[1:-1], ci = None)
plt.xlabel('Subject')
plt.ylabel('PM accuracy')
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='subj', y= 'og_acc', hue= 'block', data=all_df_combined, palette=my_pal, ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
plt.legend(title = 'Blocks',  
	bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_bysubj_compare_ogacc.png', dpi = 600)
plt.close() 
 


#allTogether_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'block']
ax = sea.violinplot(x='block', y = 'meanTrial_rt', hue = 'version', data=all_df_combined, palette = my_pal, cut = 0, split = True)
plt.xlabel('Block')
plt.ylabel('Reaction time (s)')
ax.legend(title = 'Version')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_rt.png', dpi = 600)
plt.close()

# PM Accuracy
ax = sea.barplot(x='block', y= 'pm_acc', hue = 'version', data=all_df_combined_minusBase, palette=my_pal, ci = None)
plt.xlabel('Block')
plt.ylabel('PM accuracy')
ax.legend(title = 'Version')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_pmacc.png', dpi = 600)
plt.close()

# OG Accuracy
ax = sea.barplot(x='block', y= 'og_acc', hue = 'version', data=all_df_combined, palette=my_pal, ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
ax.legend(title = 'Version')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_ogacc.png', dpi = 600)
plt.close() 


#### ALL TOGETHER - just v1.1 - blocks collapsed
#allTogether_df.columns = ['subj', 'pm_acc', 'meanTrial_rt', 'block']

all_v11_df =  all_df_combined[all_df_combined.version == 'vers1.1']
all_df_combined_minusBase_v11 = all_v11_df[all_v11_df.block != 'Baseline']

ax = sea.violinplot(x='block', y = 'meanTrial_rt', data=all_v11_df, #palette = my_pal,#
	cut = 0)
plt.xlabel('Block')
plt.ylabel('Reaction time (s)')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_rt_V1.1.png', dpi = 600)
plt.close()


# PM Accuracy
ax = sea.barplot(x='block', y= 'pm_acc', data=all_df_combined_minusBase_v11)
plt.xlabel('Block')
plt.ylabel('PM accuracy')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_pmacc_v1.1.png', dpi = 600)
plt.close() 

# OG Accuracy
ax = sea.barplot(x='block', y= 'og_acc', data=all_v11_df)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_ogacc_V1.1.png', dpi = 600)
plt.close() 

###########



### PICK UP HERE ###


#### PM cost

base_cost = (baseline_df.groupby('subj')['meanTrial_rt'].mean())

maintain1_cost = block_maintain1_df.groupby(['subj', 'version'])['meanTrial_rt'].mean()
maintain2_cost = block_maintain2_df.groupby(['subj', 'version'])['meanTrial_rt'].mean()

maintain1_cost_PM = maintain1_cost - base_cost
maintain1_cost_PM = maintain1_cost_PM.to_frame().reset_index()
maintain1_cost_PM['block'] = 'Maintain 1'

maintain2_cost_PM = maintain2_cost - base_cost
maintain2_cost_PM = maintain2_cost_PM.to_frame().reset_index()
maintain2_cost_PM['block'] = 'Maintain 2'

monitor1_cost = block_monitor1_df.groupby(['subj', 'version'])['meanTrial_rt'].mean()
monitor2_cost = block_monitor2_df.groupby(['subj', 'version'])['meanTrial_rt'].mean()

monitor1_cost_PM = monitor1_cost - base_cost
monitor1_cost_PM = monitor1_cost_PM.to_frame().reset_index()
monitor1_cost_PM['block'] = 'Monitor 1'

monitor2_cost_PM = monitor2_cost - base_cost
monitor2_cost_PM = monitor2_cost_PM.to_frame().reset_index()
monitor2_cost_PM['block'] = 'Monitor 2'

mnm1_cost = block_mnm1_df.groupby(['subj', 'version'])['meanTrial_rt'].mean()
mnm2_cost = block_mnm2_df.groupby(['subj', 'version'])['meanTrial_rt'].mean()

mnm1_cost_PM = mnm1_cost - base_cost
mnm1_cost_PM = mnm1_cost_PM.to_frame().reset_index()
mnm1_cost_PM['block'] = 'M&M 1'

mnm2_cost_PM = mnm2_cost - base_cost
mnm2_cost_PM = mnm2_cost_PM.to_frame().reset_index()
mnm2_cost_PM['block'] = 'M&M 2'

pmCost = pd.concat([maintain1_cost_PM, maintain2_cost_PM, 
		  monitor1_cost_PM, monitor2_cost_PM, 
		  mnm1_cost_PM, mnm2_cost_PM], axis = 0)

pmCost_v11 = pmCost[pmCost.version=='vers1.1']

# PM cost
ax = sea.barplot(x='block', y= 'meanTrial_rt', data=pmCost_v11)
plt.xlabel('Block')
plt.ylabel('PM cost (s)')
ax.tick_params(axis='x', labelsize=7)
sea.despine()
plt.savefig(FIGURE_PATH + 'all_together_compare_PMCOST.png', dpi = 600)
plt.close()



blockRTs = all_df_combined.groupby(['subj', 'block'])['meanTrial_rt'].mean()
baseline_df_v11 = all_df_combined[all_df_combined.block == 'Baseline']

maintain_df_v11 = all_df_combined[all_df_combined.block == 'Maintain']
monitor_df_v11 = all_df_combined[all_df_combined.block == 'Monitor']
mnm_df_v11 = all_df_combined[all_df_combined.block == 'M&M']

#####################################
############  Figures  ##############
#####################################








