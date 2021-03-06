##################################
##################################
######### monitain v1.0 ##########
######## Katie Hedgpeth ##########
######## February 2019 ###########
##################################
##################################


import random
import numpy as np 
import pandas as pd
import os
import sys
import pprint 
import argparse
from psychopy import visual, event, core
from itertools import product, compress


## Thank yoooouuu Remy
SCREENS = {
    'animal':      dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
    'beauregard':  dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
    'camilla':     dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
    'scooter':     dict(distance_cm= 67,width_cm=28.5,pixel_dims=[1440, 900]),
    'snuffy':      dict(distance_cm= 67,width_cm=59.3,pixel_dims=[2560,1440]),
    'swedishchef': dict(distance_cm= 67,width_cm=33.0,pixel_dims=[1440, 900]),
    'alice':       dict(distance_cm= 67,width_cm=28.5,pixel_dims=[1440, 900]),
}

parser = argparse.ArgumentParser(description="Monitain experimental display")
parser.add_argument('--subj', default='s999', type=str, help='sXXX format')
parser.add_argument('--scrn', default='animal', type=str, choices=SCREENS.keys(), help = 'computer used for experiment')
args = parser.parse_args()

subj = args.subj
scrn = args.scrn

# Put .txt files into dataframes
words_df = pd.read_table("words.txt", header=-1)
words_df = words_df.rename(columns={0:'stimuli'})
words_df['type'] = 1

nonwords_df = pd.read_table("nonwords.txt", header=-1)
nonwords_df = nonwords_df = nonwords_df.rename(columns={0:'stimuli'})
nonwords_df['type'] = 2

stimCombine = [words_df, nonwords_df]
ogStims_df = pd.concat(stimCombine, ignore_index=True)



#Orientations


####################################
############ Parameters ############
####################################

## Check to see if file exists
data_path = "monitain_v1_" + str(subj_id)
data_path_exists = os.path.exists(data_path)

filename = data_path + ".csv"

if data_path_exists: 
	sys.exit("Filename " + data_path + "already exists!")

# Colors
color_white = [1,1,1]
color_black = [-1,-1,-1]
color_gray = [0,0,0]

# Timings
sec_target = 2 
sec_delay = 1 
sec_probe = 2 
sec_iti = 1

# Keys to hit 
sd_keyList = ['1', '2']

# Slack 
############ NEED TO ADD 


####################################
########## Build dataframe #########
#### for individual participant ####
####################################

# columns = ['subj_id', 'block_num', 'trial', 
# 	'ori_top', 'ori_mid', 'ori_bot', 
# 	'word_nonword', 'lexicalStim', 
# 	'correctResp', 'resp', 'acc', 
# 	'rt'] ## Add more to this! 

# n_trials_base = 106
# n_trials_maintain = 198
# n_trials_monitor = 198
# n_trials_mm = 198

# n_runs = 2

# oriData = [10, 20, 30, 40, 50, 60, 70, 80, 100,
# 	110, 120, 130, 140, 150, 160, 170] #Exclude 0/180 and 90 degrees


# trialsPerBlock_a = [1, 2, 5, 6, 
# 	8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]
# trialsPerBlock_b = [3, 4, 7, 
# 	8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]


# Set values
N_MIN_PROBES = 8
N_MAX_PROBES = 15
N_PROBES_PER_BLOCK = 20

LOWER_CATCH_TRIAL = 1
UPPER_CATCH_TRIAL = 7

N_BLOCKS = 8
probe_count_list = [] 

trials = range(N_MIN_PROBES, N_MAX_PROBES+1)


baselineTrials = 106
maintainTrials = 20
monitorTrials = 198
mnmTrials = 20

#N_TOTAL_TRIALS = N_PROBES_PER_BLOCK * N_BLOCKS
N_TOTAL_TRIALS = (baselineTrials*2) + (maintainTrials*2) + (monitorTrials*2) + (mnmTrials*2)


# Create dataframe
word_cols = ['word{:d}'.format(i+1) for i in range(N_MAX_PROBES) ]
wordCond_cols = ['word{:d}_cond'.format(i+1) for i in range(N_MAX_PROBES) ]

topTheta_cols = ['topTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]
botTheta_cols = ['botTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]

columns = ['subj', #subject id 
	'block', #block num 
	'targTheta', #angle for memory target
	'n_probes',  #num of probes in trial 
	'probeTheta_loc', #where target probe is on last probe
	'acc', #acc for target probe for trial

	]

df_columns = columns + word_cols + wordCond_cols + topTheta_cols + botTheta_cols
df_index = range(N_TOTAL_TRIALS)
df = pd.DataFrame(columns = df_columns, index = df_index)

df['subj'] = subj_id

# Break up blocks
## 1 and 8 = Baseline, trials 0-105 and 582-687
## 2 and 3 = Maintain, trials 106-125, 126-145
## 4 and 5 = Monitor, trials 146-343, 344-541
## 6 and 7 = M&M, trials 542-561, 562-581

# Set block values
df.iloc[0:106, df.columns.get_loc('block')] = 1
df.iloc[106:126, df.columns.get_loc('block')] = 2
df.iloc[126:146, df.columns.get_loc('block')] = 3
df.iloc[146:344, df.columns.get_loc('block')] = 4
df.iloc[344:542, df.columns.get_loc('block')] = 5
df.iloc[542:562, df.columns.get_loc('block')] = 6
df.iloc[562:582, df.columns.get_loc('block')] = 7
df.iloc[582:688, df.columns.get_loc('block')] = 8


for totalTrials in range(N_TOTAL_TRIALS): 
	for b1 in range(baselineTrials): 
		df['block'][totalTrials] = 1
	for main1 in range(maintainTrials): 
		df['block'][totalTrials] = 2
	for main2 in range(maintainTrials): 
		df['block'][totalTrials] = 3
	for mon1 in range(monitorTrials): 
		df['block'][totalTrials] = 4
	for mon2 in range(monitorTrials): 
		df['block'][totalTrials] = 5
	for mnm1 in range(mnmTrials): 
		df['block'][totalTrials] = 6
	for mnm2 in range(mnmTrials): 
		df['block'][totalTrials] = 7 
	for b2 in range(baselineTrials): 
		df['block'][totalTrials] = 8 


for i in range(baselineTrials):
	print i

all_targetTheta_locs = np.repeat(['top', 'bot'], N_TOTAL_TRIALS/2)
np.random.shuffle(all_targetTheta_locs)
df['probeTheta_loc'] = all_targetTheta_locs # Will the top or bottom be probed?

possible_thetas = np.linspace(0,180, 18, endpoint=False) # Possible orientations presented

#possible_thetas = possible_thetas[~np.isin(possible_thetas[0,90])]

possible_thetas = np.array([10, 20, 30, 40, 50, 60, 70, 80,
	100, 110, 120, 130, 140, 150, 160, 170])

np.random.choice(possible_thetas)


def pickTheta(x): 
	return np.random.choice(possible_thetas)

df['targTheta'] = df.targTheta.apply(pickTheta)

possible_thetas = np.linspace(0,180, 18, endpoint=False)

#possible_thetas = possible_thetas[~np.isin(possible_thetas[0,90])]

possible_thetas = [10, 20, 30, 40, 50, 60, 70, 80, 
	100, 110, 120, 130, 140, 150, 160, 170]

np.random.choice(possible_thetas)




# Is this just a repeat of below?
# catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1) # Possible catch trials

# np.random.choice(catch_range) 

# probes = range(N_MIN_PROBES, N_MAX_PROBES+1)
# probe_range = np.repeat(trials,2)

N_CATCH_PER_BLOCK = N_PROBES_PER_BLOCK - probe_range.size


# new_range = np.append(catch_range, probe_range)




# for i in range(N_BLOCKS): 
# 	np.random.shuffle(new_range)
# 	probe_count_list.append(new_range)




catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1)

# np.random.choice(catch_range)

probes = range(N_MIN_PROBES, N_MAX_PROBES+1)
probe_range = np.repeat(trials,2)

new_range = np.append(catch_range, probe_range)


probe_count_list = []
for i in range(N_BLOCKS): 
	catch_subset = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
	probe_set = np.append(catch_subset, probe_range)
	np.random.shuffle(probe_set)
	probe_count_list.append(probe_set)

np.ravel(probe_count_list).size #should equal 160 for hnow
df['n_probes'] = np.ravel(probe_count_list)

fake_word_list = []
fake_nonword_list = []
#fake_word_list = np.repeat(['cat'], 1000)
fake_word_list = [ 'cat{:d}'.format(i+1) for i in range(5000)]
fake_nonword_list = [ 'dawg{:d}'.format(i+1) for i in range(5000)]

fake_list = np.vstack([fake_word_list, fake_nonword_list])
np.random.shuffle(fake_list)

for i in range(N_TOTAL_TRIALS): 
	n_probes = df.loc[i, 'n_probes']
	probe_loc = df.loc[i, 'probeTheta_loc']
	memTarg = df.loc[i, 'targTheta']
	#possible_thetas_minusTarg = possible_thetas[possible_thetas!=memTarg]
	possible_thetas_minusTarg = list(compress(possible_thetas, (possible_thetas != memTarg)))

	for j in range(n_probes): 

		cond = np.random.choice(['word', 'nonword'])
		

		col_name = 'word{:d}'.format(j+1)
		col_name_cond = 'word{:d}_cond'.format(j+1)

		thetaTop_col = 'topTheta{:d}'.format(j+1)
		thetaBot_col = 'botTheta{:d}'.format(j+1)

		if j+1 != n_probes: 
			top_theta = np.random.choice(possible_thetas_minusTarg)
			bot_theta = np.random.choice(possible_thetas_minusTarg)
		elif j+1 == n_probes: 
			if probe_loc == 'top': 
				top_theta = memTarg
				bot_theta = np.random.choice(possible_thetas_minusTarg)
			elif probe_loc == 'bot': 
				top_theta = np.random.choice(possible_thetas_minusTarg)
				bot_theta = memTarg
		else: 
			raise Warning('Nooooooo')

		if cond == 'word':
			rand_word = fake_word_list.pop(0)
		elif cond == 'nonword':
			rand_word = fake_nonword_list.pop(0)
		else: 
			raise Warning('noooooooooooo')


		#insert into df
		df.loc[i, col_name] = rand_word
		df.loc[i, col_name_cond] = cond

		df.loc[i, thetaTop_col] = top_theta
		df.loc[i, thetaTop_col] = bot_theta

#Blocks
##0 = baseline
##1 = maintain
##2 = maintain
##3 = monitor
##4 = monitor
##5 = m&m
##6 = m&m
##7 = baseline

# df_index_base = pd.MultiIndex.from_product([range(n_runs),range(n_trials_base)], names=['block', 'trial'])
# df_base = pd.DataFrame(columns=columns, index=df_index_base, dtype=float)
# df_base = df_base.rename(index={1:7}) #baseline is blocks 0 and 7

# df_index_maintain = pd.MultiIndex.from_product([range(n_runs),range(n_trials_maintain)], names=['block', 'trial'])
# df_maintain = df_maintain = pd.DataFrame(columns=columns, index=df_index_maintain, dtype=float)
# df_maintain = df_maintain.rename(index={1:2})
# df_maintain['lexicalStim'].update(ogStims_df['stimuli'])
# df_maintain['correctResp'].update(ogStims_df['type'])

# trials_maintain_a = random.sample(trialsPerBlock_a, len(trialsPerBlock_a))
# trials_maintain_b = random.sample(trialsPerBlock_b, len(trialsPerBlock_b))

# df_index_monitor = pd.MultiIndex.from_product([range(n_runs),range(n_trials_monitor)], names=['block', 'trial'])
# df_monitor = pd.DataFrame(columns=columns, index=df_index_monitor, dtype=float)
# df_monitor = df_monitor.rename(index={0:3}) #monitor is blocks 3 and 4
# df_monitor = df_monitor.rename(index={1:4})

# trials_monitor_a = random.sample(trialsPerBlock_a, len(trialsPerBlock_a))
# trials_monitor_b = random.sample(trialsPerBlock_b, len(trialsPerBlock_b))

# df_index_mm = pd.MultiIndex.from_product([range(n_runs),range(n_trials_mm)], names=['block', 'trial'])
# df_mm = pd.DataFrame(columns=columns, index=df_index_mm, dtype=float)
# df_mm = df_mm.rename(index={0:5}) #m&m is blocks 5 and 6
# df_mm = df_mm.rename(index={1:6})

# trials_mm_a = random.sample(trialsPerBlock_a, len(trialsPerBlock_a))
# trials_mm_b = random.sample(trialsPerBlock_b, len(trialsPerBlock_b))

## SET UP ##

data = []
coded_data = []
responses = []

#df_mm = pd.DataFrame(columns=df_columns, index=, dtype=float)

#df_mm['subj_id'] = subj

# Need column for number of probes in monitoring and total trials


####################################
############# Psychopy #############
####################################

clock = core.Clock()

# Window set up
win = visual.Window(
	size=[1024,576], #Small size of screen for testing it out
	units="pix", 
	fullscr=False, #set to True when running for real
	#color=color
	)

# Grating set up 
grating = visual.GratingStim(
	win=win,
	units="pix", 
	size=[150,150], #size of box with grating in pixels
	mask = "circle"
	) 

grating_ypos = [-150, 150]  ## need to change

# Text set up 
text = visual.TextStim(
	win=win, 
	#text=, 
	color=color_black, 
	height=40.0
	)



####################################
############## Events ##############
####################################
responses = []


def ogOnly(words_df):  
	win.flip()
	win.color = color_gray
	#win.flip()
	text = visual.TextStim(
		win=win, 
		text=ogStims_df.loc[trial_i, 'stimuli'], 
		color=color_black, 
		height = 40.0)
	text.draw()
	win.flip()
	keys = event.waitKeys(maxWait=sec_probe, keyList = sd_keyList, timeStamped=clock)
	print keys
	responses.append([keys])
		#responses.append([keys[0][0], keys[0][1]])

def target(targetOri_df):
	win.color = color_white
	win.flip()	
	grating.pos = [0.0,0.0] 
	grating.ori = targetOri_df.loc[trial, 'orientation'] ## Change everytime
	grating.sf = 5.0 / 80.0
	grating.contrast = 1.0
	grating.draw()
	win.flip() 
	core.wait(sec_target)


def delay(): 
	win.color = color_gray
	win.flip()
	win.flip()
	core.wait(sec_delay)
	

def OGnPMprobe(): 
	win.flip()
	win.color = color_gray
	#text
	text = visual.TextStim(
		win=win, 
		text=ogStims_df.loc[trial, 'stimuli'], 
		color=color_black, 
		height = 40.0)
	text.draw()
	#gratings
	for i_grating in range(2): 
		grating.ori = 20 ## need to change
		grating.pos = [0,grating_ypos[i_grating]]
		grating.draw()
	win.flip()
	#response
	keys = event.waitKeys(maxWait=sec_probe, keyList = sd_keyList, timeStamped=clock)
	print keys
	responses.append([keys])

def targetProbe(): 
	win.flip()
	win.color = color_gray
	#text
	text = visual.TextStim(
		win=win, 
		text=ogStims_df.loc[trial, 'stimuli'], 
		color=color_black, 
		height = 40.0)
	text.draw()
	#gratings
	for i_grating in range(2): 
		grating.ori = 20 ## need to change
		grating.pos = [0,grating_ypos[i_grating]]
		grating.draw()
	win.flip()
	#response
	keys = event.waitKeys(maxWait=sec_probe, keyList = sd_keyList, timeStamped=clock)
	print keys
	responses.append([keys])

def iti(): 
	win.flip()
	text = visual.TextStim(
		win=win, 
		text="+", 
		color=color_black, 
		height = 40.0)
	text.draw()
	win.flip()



####################################
############ Experiment ############
####################################

## Baseline
# textList = ["apple", "boat", "glorb", "laser", "jalp", "book", "ser", "paper", "lent", "olev"]

# Make dataframe of words and nonwords
# wordData = [['apple',1], ['boat', 1], ['glorb', 2], ['laser',1], ['jalp',2], 
# 	['book',1], ['ser',2], ['paper',1], ['lenp',2], ['olev',2]]
# # 1 = word, 2 = nonword
# wordStims_df = pd.DataFrame(wordData, columns=['word', 'type'])

oriData = [10, 20, 80, 110, 30, 50, 170, 150, 20, 120]
targetOri_df = pd.DataFrame(oriData, columns = ['orientation'])

for trial_i in range(N_TOTAL_TRIALS): 

	if df.iloc[trial_i, df.columns.get_loc('block')] == 1: 
		ogOnly(ogStims_df)
		#print 'baseline ',trial_i

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 2: 
		#print 'maintain1',trial_i
		for trial in range(2): ## Change to maintain block length
			target(targetOri_df)
			delay()
			for maintain_probe in range(2): ## Change length
				ogOnly(ogStims_df)
			targetProbe()
			iti()
			print 'iti'

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 3: 
		#print 'maintain2',trial_i
		for trial in range(2): ## Change to maintain block length
			target(targetOri_df)
			delay()
			for maintain_probe in range(2): ## Change length
				ogOnly(ogStims_df)
			targetProbe()
			iti()
			print 'iti'

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 4: 
		#print 'monitor1',trial_i
		for trial in range(2): ##Change to number of trials
			for probe in range(2): ##Will range from 1 to 15
				OGnPMprobe()
			targetProbe()
			iti()

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 5: 
		#print 'monitor2',trial_i
		for trial in range(2): ##Change to number of trials
			for probe in range(2): ##Will range from 1 to 15
				OGnPMprobe()
			targetProbe()
			iti()

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 6: 
		#print 'mnm1',trial_i
		for trial in range(2): ## Change to total number of trials
			target(targetOri_df)
			print win.color
			delay()
			print win.color
			for probe in range(2): ## Will also range from 1 to 15 
				OGnPMprobe()
			targetProbe()
			iti()

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 7: 
		#print 'mnm2',trial_i
		for trial in range(2): ##Change to number of trials
			for probe in range(2): ##Will range from 1 to 15
				OGnPMprobe()
			targetProbe()
			iti()

	elif df.iloc[trial_i, df.columns.get_loc('block')] == 8: 
		#print 'baseline2',trial_i
		ogOnly(ogStims_df)

	else: 
		raise Warning('yikes')





# ## Baseline, 1 block 
# for trial in range(10): ## Change to length of baseline block once I have stims
# 	ogOnly(ogStims_df)	

# ## Maintain, 2 blocks

# for maintainBlock in range(2): 
# 	for trial in range(2): ## Change to maintain block length
# 		target(targetOri_df)
# 		delay()
# 		for maintain_probe in range(2): ## Change length
# 			ogOnly(ogStims_df)
# 		targetProbe()
# 		iti()
# 		print 'iti'

# ## Monitor, 2 blocks 
# for monitorBlock in range(2):	
# 	for trial in range(2): ##Change to number of trials
# 		for probe in range(2): ##Will range from 1 to 15
# 			OGnPMprobe()
# 		targetProbe()
# 		iti()

# ## M&M, 2 blocks
# for mnmBlock in range(2): 
# 	for trial in range(2): ## Change to total number of trials
# 		target(targetOri_df)
# 		print win.color
# 		delay()
# 		print win.color
# 		for probe in range(2): ## Will also range from 1 to 15 
# 			OGnPMprobe()
# 		targetProbe()
# 		iti()

# ## Baseline, 1 block 
# for trial in range(10): ## Change to length of baseline block once I have stims
# 	ogOnly(ogStims_df)	


## MAINTAIN ## 

###Target





grating.pos = [0.0,0.0]
grating.ori = 45.0
grating.sf = 5.0 / 80.0 #spatialFrequency = # of cycles / length in pixels 
grating.contrast = 1.0

grating.draw()

win.flip()

core.wait(sec_target)

win.color=color_gray
win.flip()
keys = event.waitKeys(timeStamped=clock)
#s key for same, d key for different

print keys
if keys[0][0] == 's':
	print 'same'
elif keys[0][0] == 'd': 
	print 'diff'
else: 
	print 'yikes'
win.close()

###OG task

# Start a loop for this

textList = ["apple", "boat", "glorb", "laser", "jalp", "book", "ser", "paper", "lent", "olev"]

for trial in range(10): 

	text = visual.TextStim(
		win=win, 
		text=textList[trial], 
		color=color_black, 
		height = 40.0)
	text.draw()
	win.flip()
	core.wait(sec_probe)
	keys = event.waitKeys(timeStamped=clock)
	print keys

	responses.append([keys[0][0], keys[0][1]])

win.flip()

core.wait(sec_probe)

# Second word just for show

text.text = "boat"
text.draw()
win.flip()
core.wait(sec_probe)

# End loop

###Target probe

grating_hpos = [-150, 150]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.draw()

text.text = "glorb"
text.draw()

win.flip()
core.wait(sec_probe)


###ITI with fixation cross 

text.text = "+"
text.draw()
win.flip()
core.wait(sec_iti)

win.close()



##############


## MONITOR ##

win = visual.Window(
	size=[1024,576], #Small size of screen for testing it out
	units="pix", 
	fullscr=False, #set to True when running for real
	color=color_gray)

grating_ori = [45.0, 135.0]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "jalp"
	text.draw()
win.flip()
core.wait(sec_probe)


grating_ori = [135.0, 15.0]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "apple"
	text.draw()
win.flip()
core.wait(sec_probe)


grating_ori = [45.0, 15.0]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "boat"
	text.draw()
win.flip()
core.wait(sec_probe)


grating_ori = [135.0, 135.0]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "glorb"
	text.draw()
win.flip()
core.wait(sec_probe)

###ITI with fixation cross 

text.text = "+"
text.draw()
win.flip()
core.wait(sec_iti)

win.close()

#############


## MAINTAIN AND MONITOR ##

win = visual.Window(
	size=[1024,576], #Small size of screen for testing it out
	units="pix", 
	fullscr=False, #set to True when running for real
	color=color_white)

## MAINTAIN ## 

###Target

grating = visual.GratingStim(
	win=win,
	units="pix", 
	size=[150,150], #size of box with grating in pixels
	mask = "circle"
	) 

grating.pos = [0.0,0.0]
grating.ori = 45.0
grating.sf = 5.0 / 80.0 #spatialFrequency = # of cycles / length in pixels 
grating.contrast = 1.0

grating.draw()

win.flip()

core.wait(sec_target)

win.color=color_gray
win.flip()

grating_ori = [15.0, 135.0]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "apple"
	text.draw()
win.flip()
core.wait(sec_probe)


grating_ori = [135.0, 15.0]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "boat"
	text.draw()
win.flip()
core.wait(sec_probe)


grating_ori = [45.0, 15.0] #TARGET

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.ori = grating_ori[i_grating]
	grating.draw()
	text.text = "glorb"
	text.draw()
win.flip()
core.wait(sec_probe)


###ITI with fixation cross 

text.text = "+"
text.draw()
win.flip()
core.wait(sec_iti)

##########################


###Waiting for a response to move on 
keys = psychopy.event.waitKeys(keyList=["space"])


win.close() #Close window

##Save output 
np.savetxt() 

