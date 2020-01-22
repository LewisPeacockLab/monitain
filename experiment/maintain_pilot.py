'''

Monitain v1.1
Pilot for maintenance only 
Katie Hedgpeth
January 2020

'''

# import all needed modules
import argparse
import csv
import glob
import itertools
import os
import pprint
import random
import requests
import sys

import numpy as np
import pandas as pd

from collections import OrderedDict
from itertools import product, compress
from sklearn.utils import shuffle
from psychopy import visual, core, event, monitors
from psychopy.iohub import launchHubServer

# screen size options
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

# get subject num from entry at command line
# default is s999
# default is camilla (testing room comp)
parser = argparse.ArgumentParser(description='Monitain experimental display')
parser.add_argument('--subj', default='s999', type=str, help='sXXX format')
parser.add_argument('--scrn', default='misspiggy_main', type=str, choices=SCREENS.keys(), 
	help='computer used for experiment')
args = parser.parse_args()

SUBJ = args.subj
SCRN = args.scrn


###
### Set up Slack 
###

# set to false because github respository is public
SLACK = False

def slack(msg):
	if SLACK: 
		payload = {'text': msg, 
		'channel': SLACK['channel'], 
		'username': SLACK['botname'], 
		'icon_emoji': SLACK['emoji']}
	try: 
		requests.post(SLACK['url'], json=payload)
	except ConnectionError: 
		print('Slack messaging failed - no internet connection')

	print(msg)


### Directories and import data ## 

DATA_PATH = 'monitain_v1.1_MAINTAIN_' + str(SUBJ)
DATA_FNAME = DATA_PATH + '.csv'
if os.path.exists(DATA_FNAME):
	sys.exit('Filename ' + DATA_FNAME + " already exists!")

# import images
images = glob.glob('stimuli/grayscale/frac_*.png')

word_file = 'stimuli/words.csv'
nonword_file = 'stimuli/nonwords.csv'
pract_word_file = 'stimuli/pract_words.csv'
pract_nonword_file = 'stimuli/pract_nonwords.csv'

def read_csv(filename):
	export_list = []
	# open and read csv
	with open(filename, 'r') as csvfile: 
		csvreader = csv.reader(csvfile)
		for row in csvreader: 
			export_list.append(row)
	# shuffle stimuli
	random.shuffle(export_list) 
	# return shuffled list 
	return export_list

# import words and nonwords
word_list = read_csv(word_file)
nonword_list = read_csv(nonword_file)
pract_word_list = read_csv(pract_word_file)
pract_nonword_list = read_csv(pract_nonword_file)




### General experiment parameters ###

N_MIN_PROBES = 8
N_MAX_PROBES = 15
N_PROBES_PER_BLOCK = 20

LOWER_CATCH_TRIAL = 1
UPPER_CATCH_TRIAL = 7

N_BLOCKS = 4

# if you want to change number of blocks of each type, change here
N_BLOCKS_MAINTAIN_1 = 1
N_BLOCKS_MAINTAIN_2 = 1
N_BLOCKS_MAINTAIN_3 = 1

N_TRIALS = range(N_MIN_PROBES, N_MAX_PROBES+1)

BASELINE_TRIALS = 106
MAINTAIN_TRIALS = 20

N_TOTAL_TRIALS = (BASELINE_TRIALS*1) + (MAINTAIN_TRIALS*3)
N_TOTAL_TRIALS_PRACT = 5 * N_BLOCKS # 5 for each baseline, maintain1, maintain2, maintain3

BLOCK_STRUCT = [106, 20, 20, 20]

# colors
color_white = [255,255,255] #[1,1,1]
color_black = [0,0,0] #[-1,-1,-1]
color_gray = [85,85,85] #[0,0,0]
color_cyan = [0,255,255] #[0,1,1]
color_green = [0,255,0] #[0,1,0]
color_red = [255,0,0] #[1,0,0]
color_blue = [0,0,255]

FRACTAL_SIZE = [128, 128] # default that images came at

KEYS_WORD = ['1', '2'] # 1 = word, 2 = nonword
KEYS_TARGET = ['3']
KEYS_NONTARGET = ['4']

TOP_POS = [0, 150]
MID_POS = [0, 0]
BOT_POS = [0, -150]

# event timings
TIMINGS = {
	'target': 		2,
	'delay': 		1, 
	'probe': 		2, 
	'iti': 			1,
}

BLOCK_ORDER = {
	'base1': 		1, 
	'maintain1': 	2, 
	'maintain2':	3, 
	'maintain3':	4,
}


###
### Master dataframe
###

# create an empty dataframe that will hold all parameters for every trial
# start by creating column names
columns = ['subj', #subject id 
	'block_num', #block number
	'block_name', #block type/name
	'targTheta', #angle for memory target
	'targTheta2', #angle for updated memory target (BLOCK 4 ONLY)
	'n_probes',  #num of probes in trial 
	'probeTheta_loc', #where target probe is on last probe
	'targOrNoTarg', #is target present on final probe?
	'acc', #acc for target probe for trial
	'pm_acc' #pm acc for trial based on target probe
	]

# column for each possible word/nonword probe
word_cols = ['word{:d}'.format(i) for i in range(N_MAX_PROBES)]

# whether stimuli is a word or nonword
wordCond_cols = ['word{:d}_cond'.format(i) for i in range(N_MAX_PROBES)]

# fractal num for top image
topTheta_cols = ['topTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]

#fractal num for bottom image
botTheta_cols = ['botTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]

df_columns = columns + word_cols + wordCond_cols + topTheta_cols + botTheta_cols
df_index = range(N_TOTAL_TRIALS)

# actually create the empty dataframe
df = pd.DataFrame(columns = df_columns, index = df_index)

## DF - Subject 
df['subj'] = SUBJ

block_num_list = list(BLOCK_ORDER.values())
block_type_list = list(BLOCK_ORDER.keys()) 

## DF - Block
# set the block number for each trial
for index, row in df.iterrows(): 
	if index in range(0,106): 
		df.iloc[index, df.columns.get_loc('block_num')] = block_num_list[0]
		df.iloc[index, df.columns.get_loc('block_name')] = block_type_list[0]  
	elif index in range(106,126): 
		df.iloc[index, df.columns.get_loc('block_num')] = block_num_list[1]
		df.iloc[index, df.columns.get_loc('block_name')] = block_type_list[1] 
	elif index in range(126,146): 
		df.iloc[index, df.columns.get_loc('block_num')] = block_num_list[2]
		df.iloc[index, df.columns.get_loc('block_name')] = block_type_list[2] 		
	elif index in range(146,166): 
		df.iloc[index, df.columns.get_loc('block_num')] = block_num_list[3]
		df.iloc[index, df.columns.get_loc('block_name')] = block_type_list[3] 

## DF - Target fractal/theta
# 20 possible thetas to pick from
# before the stimuli were called theta so I'm sticking
# with that in my code and with my naming convention
possible_thetas = np.array(range(1,21))
def pickTheta(x): 
	return np.random.choice(possible_thetas)

# apply function to df
df['targTheta'] = df.targTheta.apply(pickTheta)



## add more

## DF - Number of probes

# catch range is the number of probes that are too low to count
catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1)

# trials may range from min num of probes and max num of probes
probe_range = range(N_MIN_PROBES, N_MAX_PROBES+1) 

# create array of two of each probe num to pick from when populating 
# df for each block
# this helps ensure that you don't have a block with way more of one num of probes 
# for example, a block with almost all trials of length 8 probes
possible_probes = np.repeat(probe_range, 2)

# number of catch trials per block
N_CATCH_PER_BLOCK = N_PROBES_PER_BLOCK - possible_probes.size

# create a numpy array to hold all possible probe nums for each block 
new_range = np.append(catch_range, possible_probes)

# pick probes to populate each block type in dataframe
def pickProbes(n_blocks, catch_range, N_CATCH_PER_BLOCK):
	probe_count_list = []
	for block in range(n_blocks): 
		# randomly pick num of probes for catch trials
		catch_subset = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
		probe_set = np.append(catch_subset, possible_probes)
		np.random.shuffle(probe_set)
		probe_count_list.append(probe_set)
	probe_size = np.ravel(probe_count_list).size
	probe_ravel = np.ravel(probe_count_list) # array of num of probes for trial
	return probe_ravel

#BLOCK 1 probe num
baseline_probe_range = np.repeat([1],106)
df.iloc[0:106, df.columns.get_loc('n_probes')] = baseline_probe_range

#BLOCKS 2-4 probe num
maintainProbes1 = pickProbes(N_BLOCKS_MAINTAIN_1, catch_range, N_CATCH_PER_BLOCK)
maintainProbes2 = pickProbes(N_BLOCKS_MAINTAIN_2, catch_range, N_CATCH_PER_BLOCK)
maintainProbes3 = pickProbes(N_BLOCKS_MAINTAIN_3, catch_range, N_CATCH_PER_BLOCK)

# set probe num for each trial and each block in df
df.iloc[106:126, df.columns.get_loc('n_probes')] = maintainProbes1
df.iloc[126:146, df.columns.get_loc('n_probes')] = maintainProbes2
df.iloc[146:166, df.columns.get_loc('n_probes')] = maintainProbes3

## DF - Probe fractal/theta location 
# Will the top or bottom be probed?

def pickProbeLoc():
	# list that repeats 'top', 'bot', ..., for total num trials
	location_list = np.repeat(['top', 'bot'], N_TOTAL_TRIALS/2)
	np.random.shuffle(location_list)
	np.ravel(location_list)
	return location_list

df['probeTheta_loc'] = pickProbeLoc()

## DF - Target or no target
df['targOrNoTarg'] = np.nan

# set target present for half of maintain blocks, not present for other half
targetOutcome = np.repeat([0,1], MAINTAIN_TRIALS/2)

def targetPresent(block_start, block_end): 
	np.random.shuffle(targetOutcome)
	df.iloc[block_start:block_end, df.columns.get_loc('targOrNoTarg')] = targetOutcome

# assign target presence (0 or 1/no or yes) for each block
targetPresent(106,126)
targetPresent(126,146)
targetPresent(146,166)

## DF - wordx / nonwordx and topTheta/botTheta
# assign word/nonwords stimuli that will appear in each trial 
# based on how many probes that trial will have
# also assign which images will appear on the top or bottom

def assignTheta(probe_num, possible_thetas_minusTarg):
	top_theta = np.random.choice(possible_thetas_minusTarg)
	bot_theta = np.random.choice(possible_thetas_minusTarg)
	# reassign if top and bottom are the same
	if top_theta == bot_theta: 
		new_bot_theta = [x for x in possible_thetas_minusTarg if x not in [top_theta]]
		bot_theta = np.random.choice(new_bot_theta)
	return top_theta, bot_theta

# assign the memory target to top or bottom, based on the trial's parameters
def assignMemTarg(probe_num, possible_thetas_minusTarg):
	if probe_loc == 'top':
		top_theta = memTarg
		bot_theta = np.random.choice(possible_thetas_minusTarg)
	elif probe_loc == 'bot':
		top_theta = np.random.choice(possible_thetas_minusTarg)
		bot_theta = memTarg



for trial in range(N_TOTAL_TRIALS):
	# set variables for each trial to reference as 
	# you progress through the loop
	n_probes = df.loc[trial, 'n_probes']
	probe_loc = df.loc[trial, 'probeTheta_loc']
	memTarg = df.loc[trial, 'targTheta']
	memTarg2 = df.loc[trial, 'targTheta2']
	currentBlock = df.block[trial]
	possible_thetas_minusTarg = [x for x in possible_thetas if x not in [memTarg, memTarg2]]

	# for every trial, there are a series of probes that need
	# info updated so the experiment knows 
	# which stimuli to put on the screen for each probe
	for probe in range(n_probes):
		# assign words/nonwords for each probe
		condition = np.random.choice(['word', 'nonword']) 
		col_name = 'word{:d}'.format(probe)
		col_name_cond = 'word{:d}_cond'.format(probe)

		# set words and nonwords
		if condition == 'word':
			random_word = word_list.pop(0)
		elif condition == 'nonword':
			random_word = nonword_list.pop(0)

		df.loc[trial, col_name] = random_word
		df.loc[trial, col_name_cond] = condition


		# set top and bottom image stimuli 
		thetaTop_col = 'topTheta{:d}'.format(probe)
		thetaBot_col = 'botTheta{:d}'.format(probe)

		targOrNah = df.iloc[trial, df.columns.get_loc('targOrNoTarg')]

		# probe+1 != n_probes are probes before the final probe
		if probe+1 != n_probes: 
			top_theta, bot_theta = assignTheta(probe, possible_thetas_minusTarg)
		elif probe+1 == n_probes: 
			if targOrNah == 0: # target not present for MAINTAIn
				top_theta, bot_theta = assignTheta(probe, possible_thetas_minusTarg)
			elif targOrNah == 1: #target present for MAINTAIN
				assignMemTarg(probe, possible_thetas_minusTarg)
			else: # targOrNah = np.nan in monitor and mnm
				# REWRITE this if adding in monitor and mnm
				top_theta = np.nan
				bot_theta = np.nan
		else:
			# you shouldn't get to this point but if you do, something went wrong
			print('ISSUE')
			raise Warning('Nooooo') 

		# assign images that will appear on top and bottom to df for trial
		df.loc[trial, thetaTop_col] = top_theta
		df.loc[trial, thetaBot_col] = bot_theta

		# set up columns for resp, rt, acc based on how many 
		# probes there are for that specific trial 
		resp_probe = 'respProbe{:d}'.format(probe)
		rt_probe = 'rtProbe{:d}'.format(probe)
		acc_probe = 'probe{:d}_acc'.format(probe)

		# fill them all with nan, will be filled in when 
		# subject participates
		df.loc[trial, resp_probe] = np.nan
		df.loc[trial, rt_probe] = np.nan
		df.loc[trial, acc_probe] = np.nan

# Columns in DF that will be filled in as subjects participate: 
# acc, pm acc

### need to add: block to update on


##### COME BACK AND WORK ON THIS LATER


###
### Set up PsychPy
###

clock = core.Clock()

# monitor calibration
mon = monitors.Monitor('testMonitor')
mon.setWidth(SCREENS[SCRN]['width_cm'])
mon.setSizePix(SCREENS[SCRN]['pixel_dims'])

# window set up

fullscreen = False if SUBJ=='s999' else True # smaller screen when debugging
win = visual.Window(
	mon=mon, 
	colorSpace='rgb255',
	units='pix',
	fullscrn=fullscreen,
	)

# create a dictionary of fractals using list comprehension
image_dict = { file.split('/')[-1].split('.')[0]: visual.ImageStim(win=win, image=file) 
	for file in glob.glob("stimuli/grayscale/frac_*.png") }



### Make PsychoPy stims

## IMAGES

# images for instructions
instructImage = visual.ImageStim(
	win=win, 
	size=win.size/2,
	)

# fractal positions - top, middle, bottom
stim_fractal = visual.ImageStim(
	win=win, 
	mask='circle', 
	units='pix', 
	size=FRACTAL_SIZE,
	)

# feedback behind fractals
feedback_circle = visual.Circle(
	win=win, 
	units='pix', 
	radius=70, 
	lineColor=None, 
	fillColorSpace='rgb255',
	)

stim_top = stim_fractal
stim_top.pos = TOP_POS
feedback_top = feedback_circle
feedback_top.pos = TOP_POS

stim_mid = stim_fractal
stim_mid.pos = MID_POS
feedback_mid = feedback_circle
feedback_mid.pos = MID_POS

stim_bot = stim_fractal
stim_bot.pos = BOT_POS
feedback_bot = feedback_circle
feedback_bot.pos = BOT_POS

## TEXT 

text = visual.TextStim(
	win=win, 
	color=color_cyan, 
	colorSpace='rgb255', 
	height=40.0, 
	font='Calibri',
	)


###
### Utilitiy functions
###

def clear(): 
	clock.reset()

def resetTrial():
	text.color = color_cyan
	text.size = 40.0

def pressSpace():
	while 1: 
		for key in event.getKeys():
			if key == 'space':
				win.color = color_black
				win.flip()
				return

def drawTwoStims(trial_i, probe_n):
	topLoc = df.iloc[trial_i, df.columns.get_loc('topTheta{:d}'.format(probe_n))]
	stim_top.image = image_dict['frac_{:d}'.format(topLoc)].image
	stim_top.draw()

	botLoc = df.iloc[trial_i, df.columns.get_loc('botTheta{:d}'.format(probe_n))]
	stim_bot.image = image_dict['frac_{:d}'.format(botLoc)].image
	stim_bot.draw()


def feedback_text(trial_i, probe_n, acc):
	# set correct or incorrect in df
	df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = acc
	if acc == 1: 
		text.color = color_green
	else: 
		text.color = color_red
	text.draw()
	win.flip()

def feedback_circles(trial_i, probe_n, acc, target_present):
	# set correct or incorrect in df
	df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = acc
	if acc == 1: 
		circle_top.fillColor = color_green
		circle_bot.fillColor = color_green
		circle_top.draw()
		circle_bot.draw()
		drawTwoStims(trial_i, probe_n)
		win.flip()

def getResp(trial_i, probe_n, stimDraw, lastProbe):
	clear()
	allResp = [] # array to record all button presses made
	respRT = [] # array to hold corresponding RTS for presses
	responded = False 
	duration = TIMINGS.get('probe')

	if lastProbe: 
		getResp_targ(trial_i, probe_n, stimDraw)

	else: 
		
	# participants only have a certain amount of time to respond
		while clock.getTime() < 5: 
			stim_top.autoDraw = stimDraw # only draw if stimDraw is true  
			stim_bot.autoDraw = stimDraw

			if not responded:
				for key, rt in event.getKeys(timeStamped=clock):
					
					allResp.append(key)
					respRT.append(rt)
					firstKey = allResp[0] # record all resps but only first resp really matters
					#print(firstKey, " resp ")
					#print(respRT, "rt")
					if firstKey in KEYS_WORD: 
						stim_type = df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))]
						print(stim_type)
						# hit word for word or nonword for nonword
						if (firstKey == '1' and stim_type == 'word') or (firstKey == '2' and stim_type == 'nonword'): 
							feedback_text(trial_i, probe_n, 1) # 1 for CORRECT 
							print('correct')
						# hit word for nonword or nonword for word
						elif (firstKey == '1' and stim_type != 'word') or (firstKey == '2' and stim_type != 'nonword'): # hit nonword for word
							feedback_text(trial_i, probe_n, 0) # 0 for INCORRECT
							print('incorrect')
					# function for feedback if not baseline
					### add more if adding monitoring and mnm back in 
					# would need to handle case of response is 3, which 
					# isn't allowed for maintenance and monitoring

					else:
						# picked nothing or a key that wasn't a 1 or 2
						df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0 
						print('incorrect, not in key list')
					# record resp and rt
					df.at[trial_i, 'respProbe{:d}'.format(probe_n)] == allResp[0]
					df.at[trial_i, 'rtProbe{:d}'.format(probe_n)] == respRT[0]
	
	#else: 
	#	stim_top.autoDraw = False
	#	stim_bot.autoDraw = False
		

def getResp_targ(trial_i, probe_n, stimDraw):
	keysPossible = KEYS_TARGET + KEYS_NONTARGET 
	target_present = df.iloc[trial_i, df.columns.get_loc('targOrNoTarg')]
	# if adding monitor and mnm, add if/else statements because
	# those blocks only allow KEYS_TARGET

	if firstKey in keysPossible: 
		text.draw()

		if (firstKey == '3'): # hit target
			if (target_present == 1):
				feedback_circles(trial_i, probe_n, 1) # CORRECT
				drawTwoStims(trial_i, probe_n)
			elif (target_present == 0):
				feedback_circles(trial_i, probe_n, 0) # INCORRECT
				drawTwoStims(trial_i, probe_n)
		elif (firstKey == '4'):
			if (target_present == 0):
				feedback_circles(trial_i, probe_n, 1) # CORRECT
				drawTwoStims(trial_i, probe_n)
			elif (target_present == 1):
				feedback_circles(trial_i, probe_n, 0)
				drawTwoStims(trial_i, probe_n)
		win.flip()
		
	elif firstKey in KEYS_WORD: # picked a word when should have hit target or nontarget
		df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
		text.color = color_blue
		text.draw()
		win.flip()

	else: 
		df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0

	# set pm acc = to last probe acc
	df.iloc[trial_i, df.columns.get_loc('pm_acc')] = df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))]

def breakMessage(block): 
	# Insert message instead of PPT image so it's easier to change
	# display of block number
	breakText = 'This is the end of block {:d}. \
		\nPlease hold down the space bar to move onto the next section.'.format(block-1)

	text.text = breakText
	text.height = 40.0
	text.color = color_white
	text.draw()

	win.color = color_black
	win.flip()

	pressContinue = False
	while not pressContinue: 
		keyPress = event.waitKeys()
		if keyPress == ['space']:
			pressContinue = True
			break
	win.flip()

def presentSlides(slide): 
	instructImage.image = 'exptInstruct_v1.1/exptInstruct.{:d}.png'.format(slide)
	instructImage.draw()
	win.flip()
	pressSpace()




###
### Task functions
###

# draw target onto middle of the screen
def target(trial_i):
	win.color = color_white
	win.flip()

	# draw target onto middle of screen
	stim_mid.pos = MID_POS
	stim_mid.image = image_dict['frac_{:d}'.format(df.iloc[trial_i, 
		df.columns.get_loc('targTheta')])].image
	stim_mid.draw()
	
	win.flip()
	core.wait(TIMINGS.get['target'])

# just like target() except increased load, so 2 targets - 
# one on top and one on bottom
def target_2():
	win.color = color_white
	win.flip()

	# draw targets onto top and bottom of screen
	stim_top.pos = TOP_POS
	stim_top.image = image_dict['frac_{:d}'.format(df.iloc[trial_i, 
		df.columns.get_loc('targTheta')])].image
	stim_bot.pos = BOT_POS
	stim_bot.image = image_dict['frac_{:d}'.format(df.iloc[trial_i, 
		df.columns.get_loc('targTheta2')])].image
	stim_top.draw()
	stim_bot.draw()

	win.flip()
	core.wait(TIMINGS.get['target'])

def delay():
	win.color = color_gray
	win.flip()
	win.flip()
	core.wait(TIMINGS.get('delay'))

def ogProbe():
	win.flip()
	win.color = color_gray

	text.text = df.iloc[trial_i, df.columns.get_loc('word{:d}'.format(probe_n))]
	text.draw()

	stim_top.draw()
	stim_bot.draw()

	win.flip()
	clear()

	getResp(trial_i, probe_n, stimDraw=True)
	resetTrial()

def targetProbe(trial_i, probe_n, lastProbe):
	win.flip()
	win.color = color_gray

	text.text = df.iloc[trial_i, df.columns.get_loc('word{:d}'.format(probe_n))]
	if block_type == 'maintain*' and lastProbe: 
		text.text = ''
	text.draw()

	drawTwoStims(trial_i, probe_n)
	win.flip()
	clear()
	print(lastProbe, 'last probe')
	getResp(trial_i, probe_n, stimDraw = True, lastProbe = lastProbe)

def iti():
	# draw fixation cross onto screen for iti
	stim_top.autoDraw = False
	stim_bot.autoDraw = False
	win.flip()
	text = visual.TextStim(
		win=win,
		colorSpace='rgb255',
		text='+',
		color=color_black,
		height=40.0)

def resetTrial():
	text.color = color_cyan
	text.size = 40.0


###
### Block functions
###

def baseline(trial_i): 
	probe_n = 0
	print(trial_i)
	targetProbe(trial_i, probe_n = 0, lastProbe = False)
	resetTrial()

def maintain_1(): 

def maintain_2():

def maintain_3():

## Add below functions back if running entire experiment
#def monitor():
#def mnm():


###
### Experiment
###

block_starts = [0, 106, 126, 146]
#block_starts = [0, 106, 126, 146, 166, 186, 206, 226]
pract_starts = [106, 126, 146]

for trial_i in range(N_TOTAL_TRIALS):
	block_type = df.block_name[trial_i]
	block_num = df.block_num[trial_i]
	#print(block_type, 'block_type')

	#if trial_i in block_starts:


	## BASELINE
	if block_num == 1: 
		#print(trial_i)
		baseline(trial_i)

	## MAINTAIN 1
	elif block == 2: 
		maintain1()

	## MAINTAIN 2
	elif block == 3:
		maintain2()

	## MAINTAIN 3
	elif block == 4: 
		maintain3()

# Save output at end
df.to_csv(DATA_FNAME)

win.close()
