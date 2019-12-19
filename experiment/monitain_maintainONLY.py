##################################
##################################
######### monitain v1.1 ##########
##### MAINTAINENCE ONLY PILOT ####
######## Katie Hedgpeth ##########
######### December 2019 ##########
##################################
##################################


import random
import numpy as np 
import pandas as pd
import os
import sys
import pprint 
import argparse
import requests
import glob
import itertools
from psychopy import visual, event, core, iohub, monitors
from itertools import product, compress
from sklearn.utils import shuffle
from collections import OrderedDict



####################################
###########  PARAMETERS  ###########
####################################


## Thank yoooouuu Remy
SCREENS = {
    'animal':          dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
    'beauregard':      dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
    'camilla':         dict(distance_cm= 60,width_cm=47.3,pixel_dims=[1920,1080]),
    'scooter':         dict(distance_cm= 67,width_cm=28.5,pixel_dims=[1440,900]),
    'misspiggy_main':  dict(distance_cm= 67,width_cm=68.58,pixel_dims=[2560,1440]),
    'misspiggy_side':  dict(distance_cm= 67,width_cm=50.8,pixel_dims=[1680,1050]),
    'swedishchef':     dict(distance_cm= 67,width_cm=33.0,pixel_dims=[1440,900]),
    'alice':           dict(distance_cm= 67,width_cm=28.5,pixel_dims=[1440,900]),
}
#distance_cm is anticipated distance participant is from screen

parser = argparse.ArgumentParser(description="Monitain experimental display")
parser.add_argument('--subj', default='s999', type=str, help='sXXX format')
parser.add_argument('--scrn', default='misspiggy_main', type=str, choices=SCREENS.keys(), help = 'computer used for experiment')
args = parser.parse_args()

#io = iohub.launchHubServer(); io.clearEvents('all')
#keyboard = io.devices.keyboard

SUBJ = args.subj
SCRN = args.scrn

global DEBUG 
DEBUG = SUBJ in ['debug']


## Check to see if file exists
data_path = "monitain_v1.1_MAINTAIN" + str(SUBJ)
data_path_exists = os.path.exists(data_path)

filename = data_path + ".csv"

if data_path_exists: 
	sys.exit("Filename " + data_path + "already exists!")

#full_filename = '(LewPeaLab)/BEHAVIOR/monitain/v1/' + filename
full_filename = filename

## Set up Slack notificaitons
## SLACK CODE WOULD GO HERE

# Colors
color_white = [255,255, 255] #[1,1,1]
color_black = [0,0,0] #[-1,-1,-1]
color_gray = [85,85,85] #[0,0,0]
color_cyan = [0,255,255] #[0,1,1]
color_green = [0,255,0] #[0,1,0]
color_red = [255,0,0] #[1,0,0]
color_blue = [0,0,255]

# Timings
event_times = OrderedDict([
	('sec_target',	2), 
	('sec_delay',	1),  
	('sec_probe',	2),  
	('sec_iti', 	1)])

# Debugging mode
if DEBUG == True: 
	event_times = OrderedDict([	(event, secs/50.) for event, secs in event_times.iteritems() ])


# Keys to hit 
keyList_word = ['1', '2'] #1 = word, 2 = nonword
keyList_target = ['3']
keyList_nontarget = ['4']

# Set values
N_MIN_PROBES = 8
N_MAX_PROBES = 15
N_PROBES_PER_BLOCK = 20

LOWER_CATCH_TRIAL = 1
UPPER_CATCH_TRIAL = 7

N_BLOCKS = 4

N_BLOCKS_MAINTAIN_1 = 1
N_BLOCKS_MAINTAIN_2 = 1
N_BLOCKS_MAINTAIN_3 = 1

trials = range(N_MIN_PROBES, N_MAX_PROBES+1)

# Block lengths
baselineTrials = 106
maintainTrials = 20


N_TOTAL_TRIALS = (baselineTrials*2) + (maintainTrials*3) 
N_TOTAL_TRIALS_PRACT = 5 * 3 #5 for each maintain, monitor, mnm

blockDict = OrderedDict([
	('base1', 1), 
	('maintain1', 2), 
	('maintain2', 3), 
	('maintain3', 4), 
	])

base1 = 1
maintain1 = 2
maintain2 = 3
maintain3 = 4


practDict = OrderedDict([
	('base1_prac', base1), 
	('maintain1_prac', maintain1), 
	('maintain2_prac', maintain1), 
	('maintain3_prac', maintain1)
	])



####################################
############  STIMULI  #############
####################################


# Put .txt files into dataframes
words_df = pd.read_table("words.csv", header=None)
words_df = words_df.rename(columns={0:'stimuli'})
words_df['type'] = 1
words_df = shuffle(words_df)

nonwords_df = pd.read_table("nonwords.csv", header=None)
nonwords_df = nonwords_df.rename(columns={0:'stimuli'})
nonwords_df['type'] = 2
nonwords_df = shuffle(nonwords_df)

# Practice stimuli 
pract_words_df = pd.read_table("pract_words.csv", header=None)
pract_words_df = pract_words_df.rename(columns={0:'stimuli'})
pract_words_df['type'] = 1
pract_words_df = shuffle(pract_words_df)

pract_nonwords_df = pd.read_table("pract_nonwords.csv", header=None)
pract_nonwords_df = pract_nonwords_df.rename(columns={0:'stimuli'})
pract_nonwords_df['type'] = 2
pract_nonwords_df = shuffle(pract_nonwords_df)

# Import images
img_list = glob.glob("stimuli/grayscale/*.png")



####################################
###########  DATAFRAME  ############
####################################


columns = ['subj', #subject id 
	'block', #block 
	'targTheta', #angle for memory target
	'n_probes',  #num of probes in trial 
	'probeTheta_loc', #where target probe is on last probe
	'targOrNoTarg', #is target present on final probe?
	'acc', #acc for target probe for trial
	'pm_acc' #pm acc for trial based on target probe
	]

word_cols = ['word{:d}'.format(i) for i in range(N_MAX_PROBES)]
wordCond_cols = ['word{:d}_cond'.format(i) for i in range(N_MAX_PROBES)]

topTheta_cols = ['topTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]
botTheta_cols = ['botTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]

df_columns = columns + word_cols + wordCond_cols + topTheta_cols + botTheta_cols
df_index = range(N_TOTAL_TRIALS)
df = pd.DataFrame(columns = df_columns, index = df_index)

## Practice df 
#pract_df_index = range(N_TOTAL_TRIALS_PRACT)
pract_df = pd.DataFrame(columns = df_columns, index = df_index)

## DF - Subject 
df['subj'] = SUBJ

block_list = list(blockDict.items()) 
pract_list = list(practDict.items())
## DF - Block
for index, row in df.iterrows(): 
	if index in range(0,106): 
		df.iloc[index, df.columns.get_loc('block')] = block_list[0] 
		pract_df.iloc[index, pract_df.columns.get_loc('block')] = pract_list[0]  
	elif index in range(106,126): 
		df.iloc[index, df.columns.get_loc('block')] = block_list[1] 
		pract_df.iloc[index, pract_df.columns.get_loc('block')] = pract_list[1]
	elif index in range(126,146): 
		df.iloc[index, df.columns.get_loc('block')] = block_list[2] 		
		pract_df.iloc[index, pract_df.columns.get_loc('block')] = pract_list[2]
	elif index in range(146,166): 
		df.iloc[index, df.columns.get_loc('block')] = block_list[3] 
		pract_df.iloc[index, pract_df.columns.get_loc('block')] = pract_list[3]


#df.block[x][0] gives block type
#df.block[x][1] gives block num

#df.iloc[0:106, df.columns.get_loc('targOrNoTarg')] = np.nan
#df.iloc[226:332, df.columns.get_loc('targOrNoTarg')] = np.nan

## DF - Target fractal/theta
possible_thetas = np.array(range(1,21))
np.random.choice(possible_thetas)

def pickTheta(x): 
	return np.random.choice(possible_thetas)

df['targTheta'] = df.targTheta.apply(pickTheta)
pract_df['targTheta'] = df.targTheta.apply(pickTheta)

## DF - Number of probes
catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1)

probes = range(N_MIN_PROBES, N_MAX_PROBES+1)
probe_range = np.repeat(trials,2)

N_CATCH_PER_BLOCK = N_PROBES_PER_BLOCK - probe_range.size

new_range = np.append(catch_range, probe_range)

def pickProbes(n_blocks, catch_range, N_CATCH_PER_BLOCK):
	probe_count_list = []
	for i in range(n_blocks): 
		catch_subset = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
		probe_set = np.append(catch_subset, probe_range)
		np.random.shuffle(probe_set)
		probe_count_list.append(probe_set)
	probe_size = np.ravel(probe_count_list).size
	probe_ravel_1 = np.ravel(probe_count_list)[:probe_size//2]
	probe_ravel_2 = np.ravel(probe_count_list)[probe_size//2:]
	return (probe_ravel_1, probe_ravel_2)

# Practice probes
practProbes = np.array(range(2,16))
np.random.choice(practProbes)
def pickProbes_pract(x): 
	return np.random.choice(practProbes)
pract_df['n_probes'] = pract_df.n_probes.apply(pickProbes_pract)

#BLOCK 1 probe num
baseline_probe_range = np.repeat([1],106)
df.iloc[0:106, df.columns.get_loc('n_probes')] = baseline_probe_range


#BLOCKS 2-7 probe num
(maintainProbes1, maintainProbes2) = pickProbes(N_BLOCKS_MAINTAIN_1, catch_range, N_CATCH_PER_BLOCK)
(maintainProbes, maintainProbes2) = pickProbes(N_BLOCKS_MAINTAIN_2, catch_range, N_CATCH_PER_BLOCK)
(maintainProbes1, maintainProbes2) = pickProbes(N_BLOCKS_MAINTAIN_3, catch_range, N_CATCH_PER_BLOCK)

(maintainProbes1, maintainProbes2) = pickProbes(N_BLOCKS_MAINTAIN, catch_range, N_CATCH_PER_BLOCK)
(monitorProbes1, monitorProbes2) = pickProbes(N_BLOCKS_MON, catch_range, N_CATCH_PER_BLOCK)
(mnmProbes1, mnmProbes2) = pickProbes(N_BLOCKS_MnM, catch_range, N_CATCH_PER_BLOCK)


df.iloc[106:126, df.columns.get_loc('n_probes')] = maintainProbes1
df.iloc[126:146, df.columns.get_loc('n_probes')] = maintainProbes2
df.iloc[146:166, df.columns.get_loc('n_probes')] = maintainProbes3


## DF - Probe fractal/theta location 
# Will the top or bottom be probed?

def pickProbeLoc(x): 
	np.random.shuffle(all_targetTheta_locs)
	np.ravel(all_targetTheta_locs)
	return all_targetTheta_locs
all_targetTheta_locs = np.repeat(['top', 'bot'], N_TOTAL_TRIALS/2)
df['probeTheta_loc'] = pickProbeLoc(all_targetTheta_locs)
pract_df['probeTheta_loc'] = pickProbeLoc(all_targetTheta_locs)


## DF - Target or no target
df['targOrNoTarg'] = np.nan
pract_df['targOrNoTarg'] = np.nan

blockOther_len = len(df.iloc[106:126, df.columns.get_loc('block')])
pract_len = 5 #5 trials of practice for each condition 


# Set target present for half of maintain, not present for other half of maintain
targProb_other = np.repeat([0,1], blockOther_len/2)
pract_targProb_other = np.repeat([0,1], pract_len/2 + 1)

for block_other in range(2): 
	np.random.shuffle(targProb_other)
	np.random.shuffle(pract_targProb_other)
	pract_targProb_other = pract_targProb_other[:5]

	if BLCKSTR == 'interleaved': 
		if block_other == 0: #block 2
			df.iloc[106:126, df.columns.get_loc('targOrNoTarg')] = targProb_other
			pract_df.iloc[106:111, pract_df.columns.get_loc('targOrNoTarg')] = pract_targProb_other
		elif block_other == 1: #block 5
			df.iloc[166:186, df.columns.get_loc('targOrNoTarg')] = targProb_other
	elif blockstr == 'blocked': 
		if block_other == 0: #block2 
			df.iloc[106:126, df.columns.get_loc('targOrNoTarg')] = targProb_other
			pract_df.iloc[106:111, pract_df.columns.get_loc('targOrNoTarg')] = pract_targProb_other
		elif block_other == 1: #block 3
			df.iloc[126:146, df.columns.get_loc('targOrNoTarg')] = targProb_other

# Double check that half are ones and half are zeros
# pd.value_counts(df['targOrNoTarg'].values, sort=False) 


## Fill in dataframe with words/nonwords, fractals

# Add word and nonword stims to dataframe
word_list = list(words_df['stimuli'])
nonword_list = list(nonwords_df['stimuli'])

pract_word_list = list(pract_words_df['stimuli'])
pract_nonword_list = list(pract_nonwords_df['stimuli'])

pract_trials_maintain = list(range(106,111))
pract_trials_monitor = list(range(126,131))
pract_trials_mnm = list(range(146,151))
pract_trial_list = list(itertools.chain(pract_trials_maintain, pract_trials_monitor, pract_trials_mnm))

# Fill in the rest of df and pract_df
for dataf in range(2): 
	if dataf == 0: 
		dframe_make = df
	elif dataf == 1: 
		dframe_make = pract_df

	for i in range(N_TOTAL_TRIALS):
		#only set up for practice trials 
		if dataf == 1: 
			if i not in pract_trial_list: 
				continue

		n_probes = dframe_make.loc[i, 'n_probes']
		probe_loc = dframe_make.loc[i, 'probeTheta_loc']
		memTarg = dframe_make.loc[i, 'targTheta']
		currentBlock = dframe_make.block[i][1]
		possible_thetas_minusTarg = list(compress(possible_thetas, (possible_thetas != memTarg)))

		for j in range(n_probes): 
			
			cond = np.random.choice(['word', 'nonword']) # Assign words 	
			col_name = 'word{:d}'.format(j)
			col_name_cond = 'word{:d}_cond'.format(j)

			if cond == 'word':
				if dataf == 0: 
					rand_word = word_list.pop(0)
				elif dataf == 1: 
					rand_word = pract_word_list.pop(0)
			elif cond == 'nonword':
				if dataf == 0: 
					rand_word = nonword_list.pop(0)
				elif dataf == 1: 
					rand_word = pract_nonword_list.pop(0)
			else: 
				raise Warning('noooooooooooo')

			dframe_make.loc[i, col_name] = rand_word
			dframe_make.loc[i, col_name_cond] = cond
			
			thetaTop_col = 'topTheta{:d}'.format(j) # Assign thetas #
			thetaBot_col = 'botTheta{:d}'.format(j)

			targOrNah = dframe_make.iloc[i, dframe_make.columns.get_loc('targOrNoTarg')]

			if j+1 != n_probes: 
				top_theta = np.random.choice(possible_thetas_minusTarg)
				bot_theta = np.random.choice(possible_thetas_minusTarg)
				#if (currentBlock == 4) or (currentBlock == 5): 
				if top_theta == bot_theta: 
					newBotTheta_minusTop = list(compress(possible_thetas_minusTarg, (possible_thetas_minusTarg != top_theta)))
					bot_theta = np.random.choice(newBotTheta_minusTop)
			elif j+1 == n_probes: 
				if targOrNah == 0: #target not present for MAINTAIN
					top_theta = np.random.choice(possible_thetas_minusTarg)
					bot_theta = np.random.choice(possible_thetas_minusTarg)
					if top_theta == bot_theta: 
						bot_theta_minusSame = list(compress(possible_thetas_minusTarg, (possible_thetas_minusTarg != top_theta)))
						bot_theta = np.random.choice(bot_theta_minusSame)
				elif targOrNah == 1: #target present for MAINTAIN
					if probe_loc == 'top': 
						top_theta = memTarg
						bot_theta = np.random.choice(possible_thetas_minusTarg)
					elif probe_loc == 'bot': 
						top_theta = np.random.choice(possible_thetas_minusTarg)
						bot_theta = memTarg
				else: #targOrNah = np.nan
					if (currentBlock == monitor1) or (currentBlock == monitor2): #monitor 
						top_theta = memTarg #looking for watch so top or bot probe doesn't matter
						bot_theta = memTarg
					elif (currentBlock == mnm1) or (currentBlock == mnm2): #m&m 
						if probe_loc == 'top': 
							top_theta = memTarg
							bot_theta = np.random.choice(possible_thetas_minusTarg)
						elif probe_loc == 'bot': 
							top_theta = np.random.choice(possible_thetas_minusTarg)
							bot_theta = memTarg
					elif (currentBlock == base1) or (currentBlock == base2): #baseline
							top_theta = np.random.choice(possible_thetas)
							bot_theta = np.random.choice(possible_thetas)
					else: 
						print n_probes, 'n_probes'
						raise Warning('uh oh')
			else: 
				raise Warning('Nooooooo')	



			dframe_make.loc[i, thetaTop_col] = top_theta
			dframe_make.loc[i, thetaBot_col] = bot_theta

			# Set up resp, rt, acc columns #
			resp_probe = 'respProbe{:d}'.format(j)
			rt_probe = 'rtProbe{:d}'.format(j) # need to get it to take rt
			acc_probe = 'probe{:d}_acc'.format(j)

			dframe_make.loc[i, resp_probe] = np.nan
			dframe_make.loc[i, rt_probe] = np.nan
			dframe_make.loc[i, acc_probe] = np.nan


#Might need to remove later
df = df.astype('object')
pract_df = pract_df.astype('object')


####################################
############  PSYCHOPY  ############
####################################


clock = core.Clock()

mon = monitors.Monitor('testMonitor')
mon.setWidth(SCREENS[SCRN]['width_cm'])
mon.setSizePix(SCREENS[SCRN]['pixel_dims'])

# Window set up
win = visual.Window(
	monitor=mon,
	colorSpace = 'rgb255', 
	#size=[1024,576], #Small size of screen for testing it out
	units="pix", 
	fullscr=True, #set to True when running for real
	)

windowSize = win.size

stim_dict = { fn.split('/')[-1].split('.')[0]: visual.ImageStim(win=win, image=fn)for fn in glob.glob("stimuli/grayscale/*.png") }


# Images for instructions
instructImage = visual.ImageStim(
	win = win, 
	size = win.size/2 
	)

fractal_size = [128, 128] #default that they came at 

# Fractal positions - top, middle, bottom 
stim_top = visual.ImageStim(
	win = win, 
	#colorSpace = 'rgb255', 
	mask = "circle", 
	units = "pix", 
	pos = [0, 150], 
	size = fractal_size, 
	)

stim_mid = visual.ImageStim(
	win = win, 
	#colorSpace = 'rgb255', 
	mask = "circle", 
	units = "pix", 
	pos = [0, 0], 
	size = fractal_size
	)

stim_bot = visual.ImageStim(
	win = win, 
	#colorSpace = 'rgb255', 
	mask = "circle", 
	units = "pix", 
	pos = [0, -150], 
	size = fractal_size
	)

# Feedback behind fractals
circle_top = visual.Circle(
	win = win, 
	units = "pix", 
	radius = 70,
	pos = [0, 150], 
	lineColor = None, 
	fillColorSpace = 'rgb255'
	)

circle_bot = visual.Circle(
	win = win, 
	units = "pix", 
	radius = 70,
	pos = [0, -150], 
	lineColor = None, 
	fillColorSpace = 'rgb255'
	)

# Text set up 
text = visual.TextStim(
	win=win, 
	color=color_cyan, 
	colorSpace = 'rgb255',
	height=40.0, 
	font = 'Calibri'
	)



####################################
#######  UTILITY FUNCTIONS  ########
####################################


def wordOrNonword(trial_i, probe_n, dframe): 
	if dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'word': 
		text.text = dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}'.format(probe_n))]
	elif dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'nonword': 
		text.text = dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}'.format(probe_n))]

def twoStims(trial_i, probe_n, dframe): 
	topLoc = int(dframe.iloc[trial_i, dframe.columns.get_loc('topTheta{:d}'.format(probe_n))])
	stim_top.image = stim_dict['frac_{:d}'.format(topLoc)].image
	print 'fractalTop', stim_top.image
	stim_top.draw()

	botLoc = int(dframe.iloc[trial_i, dframe.columns.get_loc('botTheta{:d}'.format(probe_n))])
	stim_bot.image = stim_dict['frac_{:d}'.format(botLoc)].image
	print 'fractalBot', stim_bot.image
	stim_bot.draw()	

def clear(): 
	clock.reset()
	if DEBUG == False: 
		event.clearEvents()	

def pressSpace(): 
	if DEBUG == True: 
		return 
	while 1: 
		for key in event.getKeys(): 			
			if key == 'space': 
				win.color = color_black
				win.flip()
				return 
			
def getResp(trial_i, probe_n, block, dframe, stimDraw): 
	allResp = []
	responded = False
	duration = event_times['sec_probe']

	#Skip responses if debugging
	if DEBUG == True: 
		responded = True
	#accPos = df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))]

	while clock.getTime() < duration: 
		if stimDraw == True: 
			stim_top.autoDraw = True
			stim_bot.autoDraw = True
		else: 
			stim_top.autoDraw = False
			stim_top.autoDraw = False

		if responded == False : 
			for key, rt in event.getKeys(timeStamped=clock):
				allResp += key, rt				
				#responded = True 
				#print df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))]
				firstKey = allResp[0] 
				if firstKey in keyList_word: 
			
					if (firstKey == '1') and (dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'word'): #picked word, correct
						dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
						text.color = color_green
						text.draw()
						win.flip()
						print 'correct'
					elif (firstKey == '1') and (dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}_cond'.format(probe_n))] != 'word'): #picked word, incorrect
						dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
						text.color = color_red
						text.draw()
						win.flip()
						print 'incorrect'
					elif (firstKey == '2') and (dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'nonword'): #picked nonword, correct
						dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
						text.color = color_green
						text.draw()
						win.flip()
						print 'correct'
					elif (firstKey == '2') and (dframe.iloc[trial_i, dframe.columns.get_loc('word{:d}_cond'.format(probe_n))] != 'nonword'): #picked nonword, incorrect
						dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
						text.color = color_red
						text.draw()
						win.flip()
						print 'incorrect'

				elif (firstKey == '3') and ((block != base1) and (block != base2) and (block != maintain1) and (block != maintain2)): 
					dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
					#stim_top.color = color_red
					#stim_bot.color = color_red
					circle_top.fillColor = color_red
					circle_bot.fillColor = color_red
					circle_top.draw()
					circle_bot.draw()
					twoStims(trial_i, probe_n, dframe)
					text.draw()
					win.flip()

				else: #picked nothing or a key that wasn't 1 or 2
					dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0

				print 'allResp', allResp

				dframe.at[trial_i, 'respProbe{:d}'.format(probe_n)] = allResp #first key
				#df.loc[trial_i, 'respProbe{:d}'.format(probe_n)]= pd.Series(allResp)
				dframe.at[trial_i, 'rtProbe{:d}'.format(probe_n)] = allResp[1] #first rt
				print dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))], 'acc'
	else: 
		stim_top.autoDraw = False
		stim_bot.autoDraw = False
		lastProbe = False

	#print 'accPos', accPos, 'probe_n', probe_n
	#print ''
	#stims.autoDraw = False #change to false after done


def getResp_targ(trial_i, probe_n, block, dframe, stimDraw): 	
	allResp = []
	responded = False
	duration = event_times['sec_probe']

	if DEBUG == True: 
		responded = True

	while clock.getTime() < duration: 
		if stimDraw == True: 
			stim_top.autoDraw = True
			stim_bot.autoDraw = True
		else: 
			stim_top.autoDraw = False
			stim_top.autoDraw = False

		if (block == maintain1) or (block == maintain2): 
			keysPossible = (keyList_target + keyList_nontarget)
		elif (block == monitor1) or (block == monitor2) or (block == mnm1) or (block == mnm2): 
			keysPossible = (keyList_target)
		else: 
			print 'not good here'

		if responded == False : 
			for key, rt in event.getKeys(timeStamped=clock):
				allResp += key, rt			
				#responded = True 
				#print df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))]
				targNoTarg = dframe.iloc[trial_i, dframe.columns.get_loc('targOrNoTarg')]
				#accPos = df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))]
				firstKey = allResp[0]
				if firstKey in keysPossible: 
					text.draw()

					if (firstKey == '3'): 
						if ((((block == maintain1) or (block == maintain2)) and (targNoTarg == 1)) or ((block == monitor1) or (block == monitor2) or (block == mnm1) or (block == mnm2))): 
							#picked target, correct
							dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
							circle_top.fillColor = color_green
							circle_bot.fillColor = color_green
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n, dframe)
							win.flip()						
							print 'correct, target present'

						elif ((((block == maintain1) or (block == maintain2)) and (targNoTarg == 0)) or ((block == monitor1) or (block == monitor2) or (block == mnm1) or (block == mnm2))): #picked target, correct
							#picked target, incorrect
							dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
							circle_top.fillColor = color_red
							circle_bot.fillColor = color_red
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n, dframe)
							win.flip()							
							print 'incorrect, target not present'					
		
					elif (firstKey == '4'): 
						if ((((block == maintain1) or (block == maintain2)) and (targNoTarg == 0)) or ((block == monitor1) or (block == monitor2) or (block == mnm1) or (block == mnm2))):
							#picked no target, correct
							dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
							circle_top.fillColor = color_green
							circle_bot.fillColor = color_green
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n, dframe)
							win.flip()							
							print 'correct, target not present'

						elif ((((block == maintain1) or (block == maintain2)) and (targNoTarg == 1)) or ((block == monitor1) or (block == monitor2) or (block == mnm1) or (block == mnm2))):
							#picked no target, incorrect
							dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
							circle_top.fillColor = color_red
							circle_bot.fillColor = color_red
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n, dframe)
							win.flip()				
							print 'incorrect, target present'
						
					else: 
						print 'yikes'

				elif (firstKey == '1') or (firstKey == '2'): 
					dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
					text.color = color_blue
					text.draw()
					win.flip()

				else: #picked nothing or a key that wasn't 1 or 2
					dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0	
					print 'missed'

				dframe.at[trial_i, 'respProbe{:d}'.format(probe_n)] = allResp #first key
				dframe.at[trial_i, 'rtProbe{:d}'.format(probe_n)] = allResp[1] #first rt

				print dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))], 'acc'
				dframe.iloc[trial_i, dframe.columns.get_loc('pm_acc')] = dframe.iloc[trial_i, dframe.columns.get_loc('probe{:d}_acc'.format(probe_n))]
	else: 
		stim_top.autoDraw = False
		stim_bot.autoDraw = False

	#stims.autoDraw = False #change to false after done

def resetTrial(): 
	text.color = color_cyan
	text.size = 40.0
	#stim_top.color = color_white
	#stim_mid.color = color_white
	#stim_bot.color = color_white

def breakMessage(block):

	breakText = "This is the end of block {:d}. \
	\nPlease hold down the space bar to move onto the next section.".format(block-1)

	text.text = breakText
	text.height=40.0
	text.color = color_white
	text.draw()
	win.color = color_black
	win.flip()

	pressContinue = False
	while pressContinue == False: 
		if DEBUG == False:
			keyPress = event.waitKeys()

			if keyPress == ['space']: 
				pressContinue = True
				break 
		else: 
			break

	win.flip()

def practiceEnd(): 

	practiceText = "This is the end of your practice trials. \
	\nPlease hold down the space bar to start the next section."

	text.text = practiceText
	text.height=40.0
	text.color = color_white
	text.draw()
	win.color = color_black
	win.flip()

	pressContinue = False
	while pressContinue == False: 
		if DEBUG == False:
			keyPress = event.waitKeys()

			if keyPress == ['space']: 
				pressContinue = True
				break 
		else: 
			break

	win.flip()

def presentSlides(slide): 
	instructImage.image = 'exptInstruct_v1.1/exptInstruct.{:d}.png'.format(slide)
	instructImage.draw()
	win.flip()
	pressSpace()

def instructionSlides(block_type): 
	if (block_type is 'base1'): 
		for slide in range(1,6): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		text.text = "Press space to begin"
		text.color = color_white
		text.draw()
		win.flip()
		pressSpace()

	elif (block_type is 'maintain1') or (block_type is 'maintain2'):
	#if trial_i == (block_starts[0] or block_starts[3]): #before maintaining
		if block_type == 'maintain2':
			presentSlides(19)
		for slide in range(6,11): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		if block_type == 'maintain1': 
			text.text = "Press space to begin practice trials"
		elif block_type == 'maintain2': 
			text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()
	
	elif (block_type is 'monitor1') or (block_type is 'monitor2'):
		for slide in range(11,14): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		if block_type == 'monitor1': 
			text.text = "Press space to begin practice trials"
		elif block_type == 'monitor2': 
			text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()

	elif (block_type is 'mnm1') or (block_type is 'mnm2'):
		for slide in range(14,19): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		if block_type == 'mnm1': 
			text.text = "Press space to begin practice trials"
		elif block_type == 'mnm2': 
			text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()

	elif (block_type is 'base2'): 
		for slide in range(20,22): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()		

## SLACK FUNCTION WOULD GO HERE

####################################
 

# def ogOnly(trial_i, probe_n, dframe): 
# 	#print 'og probe', probe_n
# 	win.flip()
# 	win.color = color_gray
# 	wordOrNonword(trial_i, probe_n, dframe)
# 	text.draw()
# 	twoStims(trial_i, probe_n, dframe)
# 	win.flip()
# 	clear()
# 	getResp(trial_i, probe_n, block, dframe, stimDraw = False)
# 	resetTrial()

def target(trial_i, dframe):
	win.color = color_white
	win.flip()	
	stim_mid.pos = [0.0,0.0] 
	stim_mid.image = stim_dict['frac_{:d}'.format(dframe.iloc[trial_i, dframe.columns.get_loc('targTheta')])].image ## Change everytime
	print stim_mid.ori 
	#stim_mid.sf = 5.0 / 80.0
	#stim_mid.contrast = 1.0
	stim_mid.draw()
	win.flip() 
	core.wait(event_times['sec_target'])

def delay(): 
	win.color = color_gray
	win.flip()
	win.flip()
	core.wait(event_times['sec_delay'])	

def OGnPMprobe(trial_i, probe_n, dframe): 
	win.flip()
	win.color = color_gray
	wordOrNonword(trial_i, probe_n, dframe)
	text.draw()
	stim_top.draw()
	stim_bot.draw()
	win.flip()
	clear()
	getResp(trial_i, probe_n, block, dframe, stimDraw = True)
	resetTrial()

def targetProbe(trial_i, probe_n, block, dframe, lastProbe): 
	#print 'target probe',probe_n
	win.flip()
	win.color = color_gray
	wordOrNonword(trial_i, probe_n, dframe)
	if ((block == maintain1) or (block == maintain2)) and (lastProbe == True): 
		text.text = ''	
	text.draw()
	twoStims(trial_i, probe_n, dframe)
	win.flip()
	clear()
	if lastProbe == True: 
		getResp_targ(trial_i, probe_n, block, dframe, stimDraw = True)
	elif lastProbe == False: 
		getResp(trial_i, probe_n, block, dframe, stimDraw = True)
	lastProbe = False
	resetTrial()

def iti(): 
	stim_top.autoDraw = False
	stim_bot.autoDraw = False
	win.flip()
	text = visual.TextStim(
		win=win, 
		colorSpace = 'rgb255',
		text="+", 
		color=color_black, 
		height = 40.0)
	clear()
	duration = event_times['sec_iti']
	while clock.getTime() < duration: 
		text.draw()
		win.flip()


####################################
########  EVENT FUNCTIONS  #########
####################################


def baseline(trial_i, block, dframe): 
	probe_n = 0
 	targetProbe(trial_i, probe_n, block, dframe, lastProbe = False)
 	resetTrial()

def maintain(trial_i, block, dframe): 
 	target(trial_i, dframe)
	delay()
	probeInTrial = dframe.iloc[trial_i, dframe.columns.get_loc('n_probes')]
	for probe_n in range(probeInTrial-1): ## Change to maintain block length 
		targetProbe(trial_i, probe_n, block, dframe, lastProbe = False)
	targetProbe(trial_i, probeInTrial-1, block, dframe, lastProbe = True) #probeInTrial is always 1 extra because starts at 1
	iti()
	resetTrial()

def monitor(trial_i, block, dframe): 
 	probeInTrial = dframe.iloc[trial_i, dframe.columns.get_loc('n_probes')]
	for probe_n in range(probeInTrial-1): ## not -1 because go through all probes as targetProbe
		targetProbe(trial_i, probe_n, block, dframe, lastProbe = False)
	targetProbe(trial_i, probeInTrial-1, block, dframe, lastProbe = True)
	iti()
	resetTrial()

def mnm(trial_i, block, dframe): 
 	target(trial_i, dframe)
	delay()
	probeInTrial = dframe.iloc[trial_i, dframe.columns.get_loc('n_probes')]
	for probe_n in range(probeInTrial-1): ## not -1 because go through all probes as targetProbe
		targetProbe(trial_i, probe_n, block, dframe, lastProbe = False)
	targetProbe(trial_i, probeInTrial-1, block, dframe, lastProbe = True)
	iti()
	resetTrial()



####################################
###########  EXPERIMENT  ###########
####################################

# Uncomment below if adding Slack back 
# Let Slack know experiment is starting
#slack_msg = 'Starting experiment'
#slackMessage(1, slack_msg)


for trial_i in range(N_TOTAL_TRIALS): 
	print trial_i
	block_starts = [0, 106, 126, 146, 166, 186, 206, 226]
	pract_starts = [106, 126, 146] 

	block_type = df.block[trial_i][0] 
	block = df.block[trial_i][1] 

	

	if trial_i in block_starts: 
		if trial_i != 0: 
			breakMessage(block) # Break before moving on
		df.to_csv(full_filename) # Save output at the end
		print block_type, 'block_type'
		win.color = color_black
		win.flip()
		instructionSlides(block_type)
		#slack_msg = 'Starting block {:d}'.format(block)
		#slackMessage(block, slack_msg)
		df.to_csv(full_filename)
	win.color = color_gray
	win.flip()

	## BASELINE
	if block == 1: 
		baseline(trial_i, block, dframe = df)

	## MAINTAIN
	elif block == 2:
	#print 'maintain1',trial_i
		if trial_i in pract_starts: 
			for pract_trial in range(5):
				current_trial = pract_trial + trial_i 
				print current_trial, 'practice'
				maintain(current_trial, block, dframe = pract_df)		
			practiceEnd()
		maintain(trial_i, block, dframe = df)

	## MONITOR
	if block == 3: 
		#print 'monitor1',trial_i 
		if trial_i in pract_starts: 
			for pract_trial in range(5):
				current_trial = pract_trial + trial_i 
				monitor(current_trial, block, dframe = pract_df)
				print current_trial, 'practice'
			practiceEnd()
		monitor(trial_i, block, dframe = df)

	## MAINTAIN & MONITOR
	elif block == 4: 
		#print 'mnm1',trial_i
		if trial_i in pract_starts: 
			for pract_trial in range(5):
				current_trial = pract_trial + trial_i 
				mnm(current_trial, block, dframe = pract_df)
				print current_trial, 'practice'
			practiceEnd()
		mnm(trial_i, block, dframe = df)

	## MAINTAIN
	elif block == 5: 
		#print 'maintain2',trial_i
		maintain(trial_i, block, dframe = df)

	## MONITOR
	elif block == 6: 
		#print 'monitor2',trial_i
		monitor(trial_i, block, dframe = df)

	## MAINTAIN & MONITOR
	elif block == 7: 
		#print 'mnm2',trial_i
		mnm(trial_i, block, dframe = df)

	# BASELINE
	elif block == 8: 
		#print 'baseline 2', trial_i
		baseline(trial_i, block, dframe = df)

	#else: 
	#	raise Warning('yikes, part 2')

# End of expt instruction slide
presentSlides(22)

#slack_msg = 'Experiment finished'
#slackMessage(block, slack_msg)

# Save output at the end
df.to_csv(full_filename)

win.close()
