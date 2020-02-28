##################################
##### lil' monitain analysis #####
##################################

#### This is a mini version of the more formal 
#### monitain analysis. This is meant to serve
#### as a starting point for learning about the 
#### analysis behind monitain and for learning 
#### more what Python can do. :) 

# first, you're going to want to import the tools/packages you 
# need to run these analyses
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

# next, you'll want to import the CSV that holds all of the participants' data
# this CSV has THOUSANDS of rows because each row is a one trial for one participant 
# so you have the 100+ trials for each participant times the number of subjects
all_df = pd.read_csv('allParticipants.csv')

## what is all_df?
# all_df is a pandas dataframe that has the following columns: 
# block', 'blockType', 'meanTrial_rt', 'og_acc', 'pm_acc', 'pm_cost', 'pm_probe_rt', 'subj']
# if you were given a dataframe and you wanted to see what columns it contained, you could do: 
# df_name.columns
all_df.columns

# all_df column breakdown
# block - returns block name and number
# blockType - returns 'Baseline', 'Maintain', 'Monitor', or 'MnM'
# meanTrial_rt - returns the average reaction time across all probes for that trial
# og_acc - returns the accuracy on the ongoing probes for that trial
# pm_acc - did the subject get the target? correct would be a 1, incorrect would be a 0
# pm_cost - How much slower was their reaction time for this vs their average RT for a baseline trial
	# we can talk more about calculating this later
# pm_probe_rt - RT on the final probe where the target was (or was not) presented
# subj - subject number

# to look at just a column, enter the dataframe['column'] or dataframe.column
# try this out with different columns to see what it does, some examples below
all_df['block']
all_df.block

all_df.og_acc
all_df['pm_acc']

# you can also do things like get the mean of all of the values in one column
all_df['pm_acc'].mean()
# the above would get the mean of ALLLLL trials


# let's visualize this data!

# we can make a bar plot to look at each subject's accuracy for each block type
# be sure to run the part below one line at a time so you can see what each line of code does
ax = sea.barplot(x='subj', y= 'pm_acc', data=all_df, ci = None)
plt.xlabel('Subject') # this allows you to name the x axis
plt.ylabel('Accuracy') # this allows you to name the y axis
sea.despine() # what did you notice this line of code does?

# after you get this to work, mess around with the colors
# you can do this by changing the input to palette
# so another way you could do the same thing with different colors would be: 
ax = sea.barplot(x='subj', y= 'pm_acc', data=all_df, palette='Blues', ci = None)

# if you want to save a figure in your current directory, run a variation of this line
# you can change the name by changing what's inside the string 'figure_x.png'
# make sure to give each image a different name or it will be saved over
plt.savefig('pm_acc_bySubj_bar.png', dpi = 600)

# if you want to close the figure you're working on and start creating a new one,
# run this line
plt.close()



# let's use the same data and start making violin plots instead 
ax = sea.violinplot(x='subj', y= 'pm_acc', data=all_df, ci = None)
plt.xlabel('Subject') # this allows you to name the x axis
plt.ylabel('Accuracy') # this allows you to name the y axis

plt.savefig('pm_acc_bySubj_violin.png', dpi = 600)
plt.close()

# at this point, I would look up the difference between a bar and violin plot
# what does a violin plot show that a bar plot doesn't? 

# let's get og acc by subject
# this is the ongoing accuracy on the word/nonword probes in the middle of every trial
ax = sea.barplot(x='subj', y= 'og_acc', data=all_df, palette='Spectral', ci = None)
plt.xlabel('Subject')
plt.ylabel('OG accuracy')
sea.despine()
plt.savefig('og_acc_bySubj_bar.png', dpi = 600)
plt.close()
# practice with the above by changing things like label names or the palette
# google 'color palettes in seaborn' to try to find some more color palettes to try

# next let's look at reaction time by subject with violins
ax = sea.violinplot(x='subj', y = 'meanTrial_rt', data=all_df, cut = 0)
# Cut = 0 so range is limited to observed data
# practice by picking a color palette and adding that to the code
plt.xlabel('Subject')
plt.ylabel('Reaction time (s)')
sea.despine()
plt.savefig('rt_bySubj_violin.png', dpi = 600)
plt.close()

# so far, all of the above have averaged all trials for a single subject together
# however, it would be more useful for us to look at this by block type
# for example, we can look at og accuracy in maintain vs og accuracy in monitor
# to do this, it would be more useful to group everyone together instead of doing by subject

# we can do this by not plotting subject on the x axis
# instead, block type will by on the x axis
# to practice looking at the data different ways, 
# you can change out which column is x and which is y 
# so change x='blockType' to x=something else and the same for y

# so let's plot PM accuracy by block type now

ax = sea.barplot(x='blockType', y= 'pm_acc', data=all_df) 
plt.xlabel('Block type')
plt.ylabel('Accuracy')
plt.ylim(0,1.0) # this puts a limit on how high the y axis can go 
				# accuracy can never be more than 100% so that's why we've capped it here
sea.despine()
plt.savefig('pm_acc_byBlock_bar.png', dpi = 600) # you can always change the file name to whatever you want
plt.close()

# OG accuracy by block type
ax = sea.barplot(x='blockType', y= 'og_acc', data=all_df)
plt.xlabel('Block type')
plt.ylabel('OG accuracy')
plt.ylim(0,1.0)
sea.despine()
plt.savefig('og_acc_byBlock_bar.png', dpi = 600)
plt.close() 

# now to practice and try on your own, using the above for reference
# Q1: how would you plot RT by block type?
# Q2: how would you plot PM cost by block type?

# insert Q1 code here


# insert Q2 code here

