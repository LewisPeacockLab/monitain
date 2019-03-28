##################################
##################################
######### monitain v1.0 ##########
######## Katie Hedgpeth ##########
######## March 2019 ###########
##################################
##################################


####### PRACTICE STUFFS
subj = 101
####### 




import random
import numpy as np 
import pandas as pd
import os
import sys
import pprint 
import argparse
from psychopy import visual, event, core, iohub
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

io = iohub.launchHubServer(); io.clearEvents('all')
keyboard = io.devices.keyboard

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


####################################
############ Parameters ############
####################################

## Check to see if file exists
data_path = "monitain_v1_" + str(subj)
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
monitorTrials = 20 
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
	'acc' #acc for target probe for trial
	]

df_columns = columns + word_cols + wordCond_cols + topTheta_cols + botTheta_cols
df_index = range(N_TOTAL_TRIALS)
df = pd.DataFrame(columns = df_columns, index = df_index)

df['subj'] = subj

# Break up blocks
## 1 and 8 = Baseline, trials 0-105 and 582-687
## 2 and 3 = Maintain, trials 106-125, 126-145
## 4 and 5 = Monitor, trials 146-343, 344-541
## 6 and 7 = M&M, trials 542-561, 562-581

# Set block values
df.iloc[0:106, df.columns.get_loc('block')] = 1
df.iloc[106:126, df.columns.get_loc('block')] = 2
df.iloc[126:146, df.columns.get_loc('block')] = 3
df.iloc[146:166, df.columns.get_loc('block')] = 4
df.iloc[166:186, df.columns.get_loc('block')] = 5
df.iloc[186:206, df.columns.get_loc('block')] = 6
df.iloc[206:226, df.columns.get_loc('block')] = 7
df.iloc[226:332, df.columns.get_loc('block')] = 8



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

possible_thetas = [10, 20, 30, 40, 50, 60, 70, 80, 
	100, 110, 120, 130, 140, 150, 160, 170]

np.random.choice(possible_thetas)



catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1)


probes = range(N_MIN_PROBES, N_MAX_PROBES+1)
probe_range = np.repeat(trials,2)

N_CATCH_PER_BLOCK = N_PROBES_PER_BLOCK - probe_range.size

new_range = np.append(catch_range, probe_range)

N_BLOCKS_MAINTAIN = 2
N_BLOCKS_MON = 2
N_BLOCKS_MnM = 2

#Baseline 1 probe num
baseline_probe_range = np.repeat([1],106)
df.iloc[0:106, df.columns.get_loc('n_probes')] = baseline_probe_range

#Maintain probe num 
probe_count_list_main = []
for i in range(N_BLOCKS_MAINTAIN): 
	catch_subset_main = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
	probe_set_main = np.append(catch_subset_main, probe_range)
	np.random.shuffle(probe_set_main)
	probe_count_list_main.append(probe_set_main)

np.ravel(probe_count_list_main).size 
#df['n_probes'] = np.ravel(probe_count_list)
df.iloc[106:146, df.columns.get_loc('n_probes')] = np.ravel(probe_count_list_main) ## need to change this so it's not just setting subset in middle

#Monitor probe num 
probe_count_list_mon = []
for i in range(N_BLOCKS_MON): 
	catch_subset_mon = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
	probe_set_mon = np.append(catch_subset_mon, probe_range)
	np.random.shuffle(probe_set_mon)
	probe_count_list_mon.append(probe_set_mon)

np.ravel(probe_count_list_mon).size #should equal 160 for hnow
#df['n_probes'] = np.ravel(probe_count_list)
df.iloc[146:186, df.columns.get_loc('n_probes')] = np.ravel(probe_count_list_mon)

#Maintain&monitor probe num 
probe_count_list_mnm = []
for i in range(N_BLOCKS_MnM): 
	catch_subset_mnm = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
	probe_set_mnm = np.append(catch_subset_mnm, probe_range)
	np.random.shuffle(probe_set_mnm)
	probe_count_list_mnm.append(probe_set_mnm)

np.ravel(probe_count_list_mnm).size #should equal 160 for hnow
#df['n_probes'] = np.ravel(probe_count_list)
df.iloc[186:226, df.columns.get_loc('n_probes')] = np.ravel(probe_count_list_mnm)

#Baseline 2 probe num
df.iloc[226:332, df.columns.get_loc('n_probes')] = baseline_probe_range




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

		resp_probe = 'respProbe{:d}'.format(j+1)
		rt_probe = 'rtProbe{:d}'.format(j+1) # need to get it to take rt

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
############## Set up ##############
####################################


def collectResp(duration):
	event.clearEvents()
	clock.reset()
	while clock.getTime() < duration: 
		for key, rt in event.getKeys(keyList=sd_keyList,timeStamped=clock):
			print key, rt	
			return key, rt
	




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
	#keys = event.waitKeys(maxWait=sec_probe, keyList = sd_keyList, timeStamped=clock)
	#keys = event.getKeys(keyList = sd_keyList, timeStamped=clock)
	key, rt = collectResp(2)
	#print key
		#keys = event.getKeys(keyList = sd_keyList, timeStamped=clock)
	#print keys
	responses.append([key, rt])
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
