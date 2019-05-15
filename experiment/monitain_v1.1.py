##################################
##################################
######### monitain v1.1 ##########
######## Katie Hedgpeth ##########
########### May 2019 #############
##################################
##################################

### Error? Try this
#pip install opencv-python

import random
import numpy as np 
import pandas as pd
import os
import sys
import pprint 
import argparse
import requests
import glob
import cv2
from psychopy import visual, event, core, iohub, monitors
from itertools import product, compress
from sklearn.utils import shuffle
from collections import OrderedDict
from PIL import Image


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

subj = args.subj
scrn = args.scrn

global debug 
debug = subj in ['debug']

# Put .txt files into dataframes
words_df = pd.read_table("words.csv", header=-1)
words_df = words_df.rename(columns={0:'stimuli'})
words_df['type'] = 1

words_df = shuffle(words_df)

nonwords_df = pd.read_table("nonwords.csv", header=-1)
nonwords_df = nonwords_df.rename(columns={0:'stimuli'})
nonwords_df['type'] = 2

nonwords_df = shuffle(nonwords_df)

#stimCombine = [words_df, nonwords_df]
#ogStims_df = pd.concat(stimCombine, ignore_index=True) # change name later when bigger set 
#ogStims_df = shuffle(ogStims_df)
#ogStims_df = ogStims_df.reset_index(drop=True)



####################################
############ Parameters ############
####################################

## Check to see if file exists
data_path = "monitain_v1_" + str(subj)
data_path_exists = os.path.exists(data_path)

filename = data_path + ".csv"

if data_path_exists: 
	sys.exit("Filename " + data_path + "already exists!")

#full_filename = '(LewPeaLab)/BEHAVIOR/monitain/v1/' + filename
full_filename = filename

## Set up Slack notificaitons
SLACK = dict(
	channel = '#katieshooks', 
	botname = '{:s}'.format(subj), 
	emoji = ':person_climbing:', 
	url = 'https://hooks.slack.com/services/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'
	)

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

# Import stims
# x_data = []
# files = glob.glob("stimuli/grayscale/*.png") 
# for myFile in files: 
# 	image = cv2.imread(myFile)
# 	x_data.append(image)



# for imagename in os.listdir("stimuli/grayscale/"): 
# 	if imagename.endswith(".png"): 
# 		imagename[-6:-4] = Image.open(imagename)

# df['topFracFname'] = [ img_list[]]
# imgStims = {
# 	'top': visual.ImageStim(win, pos = [0, +10]), 
# 	'mid': visual.ImageStim(win, pos = [0, 0]),
# 	'bot': visual.ImageStim(win, pos = [0, -10])
# }
# Call by imgStims['top'], etc

# Debugging mode
if debug == True: 
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

N_BLOCKS = 8
probe_count_list = [] 

trials = range(N_MIN_PROBES, N_MAX_PROBES+1)

# Block lengths
baselineTrials = 106
maintainTrials = 20
monitorTrials = 20 
mnmTrials = 20

N_TOTAL_TRIALS = (baselineTrials*2) + (maintainTrials*2) + (monitorTrials*2) + (mnmTrials*2)


# Create dataframe
word_cols = ['word{:d}'.format(i) for i in range(N_MAX_PROBES)]
wordCond_cols = ['word{:d}_cond'.format(i) for i in range(N_MAX_PROBES)]

topTheta_cols = ['topTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]
botTheta_cols = ['botTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]

columns = ['subj', #subject id 
	'block', #block num 
	'targTheta', #angle for memory target
	'n_probes',  #num of probes in trial 
	'probeTheta_loc', #where target probe is on last probe
	'targOrNoTarg', #is target present on final probe?
	'acc', #acc for target probe for trial
	'pm_acc' #pm acc for trial based on target probe
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
df.iloc[0:106, df.columns.get_loc('block')] = 1 #blocks 1&8 are length 106
df.iloc[106:126, df.columns.get_loc('block')] = 2 #blocks 2-7 are length 20
df.iloc[126:146, df.columns.get_loc('block')] = 3
df.iloc[146:166, df.columns.get_loc('block')] = 4
df.iloc[166:186, df.columns.get_loc('block')] = 5
df.iloc[186:206, df.columns.get_loc('block')] = 6
df.iloc[206:226, df.columns.get_loc('block')] = 7
df.iloc[226:332, df.columns.get_loc('block')] = 8

df.iloc[0:106, df.columns.get_loc('targOrNoTarg')] = np.nan
df.iloc[146:332, df.columns.get_loc('targOrNoTarg')] = np.nan

blockOther_len = len(df.iloc[106:126, df.columns.get_loc('block')])

## MAINTAIN 
# Set target present for half of maintain, not present for other half of maintain
targProb_other = np.repeat([0,1], blockOther_len/2)
for block_other in range(2): 
	np.random.shuffle(targProb_other)
	if block_other == 0: #block 2
		df.iloc[106:126, df.columns.get_loc('targOrNoTarg')] = targProb_other
	elif block_other == 1: #block 3
		df.iloc[126:146, df.columns.get_loc('targOrNoTarg')] = targProb_other

# Double check that half are ones and half are zeros
# pd.value_counts(df['targOrNoTarg'].values, sort=False) 

# Will the top or bottom be probed?
all_targetTheta_locs = np.repeat(['top', 'bot'], N_TOTAL_TRIALS/2)
np.random.shuffle(all_targetTheta_locs)
df['probeTheta_loc'] = all_targetTheta_locs 

possible_thetas = np.linspace(0,180, 18, endpoint=False) # Possible orientations presented

#possible_thetas = possible_thetas[~np.isin(possible_thetas[0,90])]

# possible_thetas = np.array([10, 20, 30, 40, 50, 60, 70, 80,
# 	100, 110, 120, 130, 140, 150, 160, 170])

possible_thetas = np.array(range(1,21))

np.random.choice(possible_thetas)


def pickTheta(x): 
	return np.random.choice(possible_thetas)


df['targTheta'] = df.targTheta.apply(pickTheta)

# possible_thetas = np.linspace(0,180, 18, endpoint=False)

# possible_thetas = [10, 20, 30, 40, 50, 60, 70, 80, 
# 	100, 110, 120, 130, 140, 150, 160, 170]

# np.random.choice(possible_thetas)

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

# Add word and nonword stims to dataframe
word_list = list(words_df['stimuli'])
nonword_list = list(nonwords_df['stimuli'])

for i in range(N_TOTAL_TRIALS): 
	n_probes = df.loc[i, 'n_probes']
	probe_loc = df.loc[i, 'probeTheta_loc']
	memTarg = df.loc[i, 'targTheta']
	currentBlock = df.loc[i, 'block']
	#possible_thetas_minusTarg = possible_thetas[possible_thetas!=memTarg]
	possible_thetas_minusTarg = list(compress(possible_thetas, (possible_thetas != memTarg)))

	for j in range(n_probes): 

		# Assign words #
		cond = np.random.choice(['word', 'nonword'])
		
		col_name = 'word{:d}'.format(j)
		col_name_cond = 'word{:d}_cond'.format(j)

		if cond == 'word':
			rand_word = word_list.pop(0)
		elif cond == 'nonword':
			rand_word = nonword_list.pop(0)
		else: 
			raise Warning('noooooooooooo')

		df.loc[i, col_name] = rand_word
		df.loc[i, col_name_cond] = cond

		# Assign thetas #
		thetaTop_col = 'topTheta{:d}'.format(j)
		thetaBot_col = 'botTheta{:d}'.format(j)

		targOrNah = df.iloc[i, df.columns.get_loc('targOrNoTarg')]

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
			elif targOrNah == 1: #target present for MAINTAIN
				if probe_loc == 'top': 
					top_theta = memTarg
					bot_theta = np.random.choice(possible_thetas_minusTarg)
				elif probe_loc == 'bot': 
					top_theta = np.random.choice(possible_thetas_minusTarg)
					bot_theta = memTarg
			else: #targOrNah = np.nan
				if (currentBlock == 4) or (currentBlock == 5): #monitor 
					top_theta = memTarg #looking for watch so top or bot probe doesn't matter
					bot_theta = memTarg
				elif (currentBlock == 6) or (currentBlock == 7): #m&m 
					if probe_loc == 'top': 
						top_theta = memTarg
						bot_theta = np.random.choice(possible_thetas_minusTarg)
					elif probe_loc == 'bot': 
						top_theta = np.random.choice(possible_thetas_minusTarg)
						bot_theta = memTarg
				elif (currentBlock == 1) or (currentBlock == 8): #baseline
						top_theta = np.random.choice(possible_thetas)
						bot_theta = np.random.choice(possible_thetas)

						#top_theta = np.nan
						#bot_theta = np.nan
				else: 
					raise Warning('uh oh')
		else: 
			raise Warning('Nooooooo')	

		df.loc[i, thetaTop_col] = top_theta
		df.loc[i, thetaBot_col] = bot_theta

		# Set up resp, rt, acc columns #
		resp_probe = 'respProbe{:d}'.format(j)
		rt_probe = 'rtProbe{:d}'.format(j) # need to get it to take rt
		acc_probe = 'probe{:d}_acc'.format(j)

		df.loc[i, resp_probe] = np.nan
		df.loc[i, rt_probe] = np.nan
		df.loc[i, acc_probe] = np.nan


#Might need to remove later
df = df.astype('object')

####################################
############# Psychopy #############
####################################

clock = core.Clock()

mon = monitors.Monitor('testMonitor')
mon.setWidth(SCREENS[scrn]['width_cm'])
mon.setSizePix(SCREENS[scrn]['pixel_dims'])

# Window set up
win = visual.Window(
	monitor=mon,
	colorSpace = 'rgb255', 
	#size=[1024,576], #Small size of screen for testing it out
	units="pix", 
	fullscr=False, #set to True when running for real
	)


# Images for instructions

windowSize = win.size

instructImage = visual.ImageStim(
	win = win, 
	size = win.size/2 
	)

# Shared stims parameters
# stims_size = [150, 150]
# stims_sf = 5.0 / 80.0
# stims_contrast = 1.0

# # stims set up 
# stim_top = visual.stimsStim(
# 	win = win,
# 	mask = "circle",
# 	units="pix", 
# 	pos = [0, 150],
# 	size=stims_size, 
# 	sf = 5.0 / 80.0,
# 	contrast = stims_contrast
# 	) 

# stim_mid = visual.stimsStim(
# 	win = win,
# 	mask = "circle",
# 	units="pix", 
# 	pos = [0, 0],
# 	size=stims_size, 
# 	sf = stims_sf,
# 	contrast = stims_contrast
# 	) 

# stim_bot = visual.stimsStim(
# 	win = win,
# 	mask = "circle",
# 	units="pix", 
# 	pos = [0,-150],
# 	size=stims_size, 
# 	sf = stims_sf,
# 	contrast = stims_contrast
# 	) 

# stims_ypos = [-150, 150]  ## need to change

fractal_size = [128, 128] #default that they came at 

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

circle_top = visual.Circle(
	win = win, 
	units = "pix", 
	radius = 70,
	pos = [0, 150], 
	colorSpace = 'rgb255')

circle_bot = visual.Circle(
	win = win, 
	units = "pix", 
	radius = 70,
	pos = [0, -150], 
	colorSpace = 'rgb255')

# Text set up 
text = visual.TextStim(
	win=win, 
	color=color_cyan, 
	colorSpace = 'rgb255',
	height=40.0, 
	font = 'Calibri'
	)

# Import images
img_list = glob.glob("stimuli/grayscale/*.png")

stim_dict = { fn.split('/')[-1].split('.')[0]: visual.ImageStim(win=win, image=fn)\
	for fn in glob.glob("stimuli/grayscale/*.png") }



####################################
############## Set up ##############
####################################

def wordOrNonword(trial_i, probe_n): 
	if df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'word': 
		text.text = df.iloc[trial_i, df.columns.get_loc('word{:d}'.format(probe_n))]
	elif df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'nonword': 
		text.text = df.iloc[trial_i, df.columns.get_loc('word{:d}'.format(probe_n))]

# def twoGratings(trial_i, probe_n): 
# 	stim_top.ori = df.iloc[trial_i, df.columns.get_loc('topTheta{:d}'.format(probe_n))]
# 	print 'stimsTop', stim_top.ori
# 	stim_top.draw()
# 	stim_bot.ori = df.iloc[trial_i, df.columns.get_loc('botTheta{:d}'.format(probe_n))]
# 	print 'stimsBot', stim_bot.ori
# 	stim_bot.draw()

def twoStims(trial_i, probe_n): 
	topLoc = int(df.iloc[trial_i, df.columns.get_loc('topTheta{:d}'.format(probe_n))])
	stim_top.image = stim_dict['frac_{:d}'.format(topLoc)].image
	print 'fractalTop', stim_top.image
	stim_top.draw()

	botLoc = int(df.iloc[trial_i, df.columns.get_loc('botTheta{:d}'.format(probe_n))])
	stim_bot.image = stim_dict['frac_{:d}'.format(botLoc)].image
	print 'fractalBot', stim_bot.image
	stim_bot.draw()	


def clear(): 
	clock.reset()
	if debug == False: 
		event.clearEvents()	

def pressSpace(): 
	if debug == True: 
		return 
	while 1: 
		for key in event.getKeys(): 			
			if key == 'space': 
				win.color = color_black
				win.flip()
				#win.flip()
				return 
			

def getResp(trial_i, probe_n, block, stimDraw): 
	allResp = []
	responded = False
	duration = event_times['sec_probe']

	#Skip responses if debugging
	if debug == True: 
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
					#text.color = color_cyan #flip text to blue if input taken
					#text.draw()
					#win.flip()
			
					if (firstKey == '1') and (df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'word'): #picked word, correct
						df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
						text.color = color_green
						text.draw()
						win.flip()
						print 'correct'
					elif (firstKey == '1') and (df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))] != 'word'): #picked word, incorrect
						df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
						text.color = color_red
						text.draw()
						win.flip()
						print 'incorrect'
					elif (firstKey == '2') and (df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))] == 'nonword'): #picked nonword, correct
						df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
						text.color = color_green
						text.draw()
						win.flip()
						print 'correct'
					elif (firstKey == '2') and (df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))] != 'nonword'): #picked nonword, incorrect
						df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
						text.color = color_red
						text.draw()
						win.flip()
						print 'incorrect'

				elif (firstKey == '3') and ((block != 1) and (block != 8) and (block != 2) and (block != 3)): 
					df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
					#stim_top.color = color_red
					#stim_bot.color = color_red
					circle_top.color = 'red'
					circle_bot.color = 'red'
					circle_top.draw()
					circle_bot.draw()
					twoStims(trial_i, probe_n)
					text.draw()
					win.flip()

				else: #picked nothing or a key that wasn't 1 or 2
					df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0

				print 'allResp', allResp

				df.at[trial_i, 'respProbe{:d}'.format(probe_n)] = allResp #first key
				#df.loc[trial_i, 'respProbe{:d}'.format(probe_n)]= pd.Series(allResp)
				df.at[trial_i, 'rtProbe{:d}'.format(probe_n)] = allResp[1] #first rt
				print df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))], 'acc'
	else: 
		stim_top.autoDraw = False
		stim_bot.autoDraw = False
		lastProbe = False

	#print 'accPos', accPos, 'probe_n', probe_n
	#print ''
	#stims.autoDraw = False #change to false after done


def getResp_targ(trial_i, probe_n, block, stimDraw): 	
	allResp = []
	responded = False
	duration = event_times['sec_probe']

	if debug == True: 
		responded = True

	while clock.getTime() < duration: 
		if stimDraw == True: 
			stim_top.autoDraw = True
			stim_bot.autoDraw = True
		else: 
			stim_top.autoDraw = False
			stim_top.autoDraw = False

		if (block == 2) or (block == 3): 
			keysPossible = (keyList_target + keyList_nontarget)
		elif (block == 4) or (block == 5) or (block == 6) or (block == 7): 
			keysPossible = (keyList_target)
		else: 
			print 'not good here'

		if responded == False : 
			for key, rt in event.getKeys(timeStamped=clock):
				allResp += key, rt
				
				#responded = True 
				#print df.iloc[trial_i, df.columns.get_loc('word{:d}_cond'.format(probe_n))]
				targNoTarg = df.iloc[trial_i, df.columns.get_loc('targOrNoTarg')]
				#accPos = df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))]
				firstKey = allResp[0]
				if firstKey in keysPossible: 
					text.draw()
					#stim_top.color = color_cyan
					#stim_bot.color = color_cyan
					#twoStims(trial_i, probe_n)
					#win.flip()

					if (firstKey == '3'): 
						if ((((block == 2) or (block == 3)) and (targNoTarg == 1)) or ((block == 4) or (block == 5) or (block == 6) or (block == 7))): 
							#picked target, correct
							df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
							circle_top.color = 'green'
							circle_bot.color = 'green'
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n)
							win.flip()						
							print 'correct, target present'
						elif ((((block == 2) or (block == 3)) and (targNoTarg == 0)) or ((block == 4) or (block == 5) or (block == 6) or (block == 7))): #picked target, correct
							#picked target, incorrect
							df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
							circle_top.color = 'red'
							circle_bot.color = 'red'
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n)
							win.flip()							
							print 'incorrect, target not present'					
		
					elif (firstKey == '4'): 
						if ((((block == 2) or (block == 3)) and (targNoTarg == 0)) or ((block == 4) or (block == 5) or (block == 6) or (block == 7))):
							#picked no target, correct
							df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 1
							circle_top.color = 'green'
							circle_bot.color = 'green'
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n)
							win.flip()							
							print 'correct, target not present'

						elif ((((block == 2) or (block == 3)) and (targNoTarg == 1)) or ((block == 4) or (block == 5) or (block == 6) or (block == 7))):
							#picked no target, incorrect
							df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
							circle_top.color = 'red'
							circle_bot.color = 'red'
							circle_top.draw()
							circle_bot.draw()
							twoStims(trial_i, probe_n)
							win.flip()				
							print 'incorrect, target present'
						
					else: 
						print 'yikes'

				elif (firstKey == '1') or (firstKey == '2'): 
					df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0
					text.color = color_blue
					text.draw()
					win.flip()

				else: #picked nothing or a key that wasn't 1 or 2
					df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))] = 0	
					#stim_top.color = color_red
					#stim_bot.color = color_red
					#twoStims(trial_i, probe_n)
					win.flip()
					print 'missed'

				df.at[trial_i, 'respProbe{:d}'.format(probe_n)] = allResp #first key
				df.at[trial_i, 'rtProbe{:d}'.format(probe_n)] = allResp[1] #first rt

				print df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))], 'acc'
				#print stim_top.color
				#print stim_bot.color
				df.iloc[trial_i, df.columns.get_loc('pm_acc')] = df.iloc[trial_i, df.columns.get_loc('probe{:d}_acc'.format(probe_n))]
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
	\nPlease hold down the space bar to move onto the next block.".format(block-1)

	text.text = breakText
	text.height=40.0
	text.color = color_white
	text.draw()
	win.color = color_black
	win.flip()

	pressContinue = False
	while pressContinue == False: 
		if debug == False:
			keyPress = event.waitKeys()

			if keyPress == ['space']: 
				pressContinue = True
				break 
		else: 
			break

	win.flip()

def presentSlides(slide): 
	instructImage.image = 'exptInstruct/exptInstruct.{:d}.png'.format(slide)
	instructImage.draw()
	win.flip()
	pressSpace()


def instructionSlides(trial_i): 
	if trial_i == block_starts[0]: #before maintaining
		for slide in range(6,11): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()
		
	elif trial_i == block_starts[2]: #before monitoring
		for slide in range(11,14): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()

	elif trial_i == block_starts[4]: #before mnm
		for slide in range(14,19): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()

	elif trial_i == block_starts[6]: #before baseline
		for slide in range(19,21): ##change later based on slides
			presentSlides(slide)
		win.color = color_gray
		win.flip()
		text.text = "Press space to begin"
		text.draw()
		win.flip()
		pressSpace()

		

def slackMessage(block, slack_msg): #thank you again, Remy
	if SLACK: 
		payload = dict(text=slack_msg, channel = SLACK['channel'], username=SLACK['botname'], icon_emoji=SLACK['emoji'])
		try: requests.post(json=payload, url=SLACK['url'])
		except ConnectionError: print('Slack messaging failed, no internet')



####################################
############## Events ##############
####################################
 

def ogOnly(trial_i, probe_n): 
	#print 'og probe', probe_n
	win.flip()
	win.color = color_gray
	wordOrNonword(trial_i, probe_n)
	text.draw()
	twoStims(trial_i, probe_n)
	win.flip()
	clear()
	getResp(trial_i, probe_n, block, stimDraw = False)
	resetTrial()


def target(trial_i):
	win.color = color_white
	win.flip()	
	stim_mid.pos = [0.0,0.0] 
	stim_mid.image = stim_dict['frac_{:d}'.format(df.iloc[trial_i, df.columns.get_loc('targTheta')])].image ## Change everytime
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

def OGnPMprobe(trial_i, probe_n): 
	win.flip()
	win.color = color_gray
	wordOrNonword(trial_i, probe_n)
	text.draw()
	stim_top.draw()
	stim_bot.draw()
	win.flip()
	clear()
	getResp(trial_i, probe_n, block, stimDraw = True)
	resetTrial()


def targetProbe(trial_i, probe_n, block, lastProbe): 
	#print 'target probe',probe_n
	win.flip()
	win.color = color_gray
	wordOrNonword(trial_i, probe_n)
	if ((block == 2) or (block == 3)) and (lastProbe == True): 
		text.text = ''	
	text.draw()
	twoStims(trial_i, probe_n)
	win.flip()
	clear()
	if lastProbe == True: 
		getResp_targ(trial_i, probe_n, block, stimDraw = True)
	elif lastProbe == False: 
		getResp(trial_i, probe_n, block, stimDraw = True)
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
############ Experiment ############
####################################

# Let Slack know experiment is starting
slack_msg = 'Starting experiment'
slackMessage(1, slack_msg)

# Starting instruction slides
win.color = color_black
win.flip()
for start_slide in range(1,6): ##change later based on slides
	presentSlides(start_slide)

win.color = color_gray
win.flip()

for trial_i in range(N_TOTAL_TRIALS): 

	block_starts = [106, 126, 146, 166, 186, 206, 226]

	block = df.iloc[trial_i, df.columns.get_loc('block')]

	if trial_i in block_starts: 
		# Break before moving on 
		breakMessage(block)
		# Save output at the end
		df.to_csv(full_filename)
		instructionSlides(trial_i)

		slack_msg = 'Starting block {:d}'.format(block)
		slackMessage(block, slack_msg)
		df.to_csv(full_filename)

	## BASELINE
	if block == 1: 
		#print 'baseline1', trial_i
	 	probe_n = 0
	 	#ogOnly(trial_i, probe_n)
	 	targetProbe(trial_i, probe_n, block, lastProbe = False)
	 	#targetProbe
	 	resetTrial()

	## MAINTAIN
	elif block == 2: 
		#print 'maintain1',trial_i
		target(trial_i)
		delay()
		probeInTrial = df.iloc[trial_i, df.columns.get_loc('n_probes')]
		for probe_n in range(probeInTrial-1): ## Change to maintain block length 
			#print 'probe',probe_n
			#ogOnly(trial_i, probe_n)
			targetProbe(trial_i, probe_n, block, lastProbe = False)
		targetProbe(trial_i, probeInTrial-1, block, lastProbe = True) #probeInTrial is always 1 extra because starts at 1
		iti()
		resetTrial()

	## MONITOR
	elif block == 3: 
		#print 'monitor1',trial_i 
		probeInTrial = df.iloc[trial_i, df.columns.get_loc('n_probes')]
		for probe_n in range(probeInTrial-1): ## not -1 because go through all probes as targetProbe
			#print 'probe', probe_n
			targetProbe(trial_i, probe_n, block, lastProbe = False)
		targetProbe(trial_i, probeInTrial-1, block, lastProbe = True)
		iti()
		resetTrial()

	## MAINTAIN & MONITOR
	elif block == 4: 
		#print 'mnm1',trial_i
		target(trial_i)
		delay()
		probeInTrial = df.iloc[trial_i, df.columns.get_loc('n_probes')]
		for probe_n in range(probeInTrial-1): ## not -1 because go through all probes as targetProbe
			#print 'probe', probe_n
			targetProbe(trial_i, probe_n, block, lastProbe = False)
		targetProbe(trial_i, probeInTrial-1, block, lastProbe = True)
		iti()
		resetTrial()

	## MAINTAIN
	elif block == 5: 
		#print 'maintain2',trial_i
		target(trial_i)
		delay()
		probeInTrial = df.iloc[trial_i, df.columns.get_loc('n_probes')]
		for probe_n in range(probeInTrial-1): ## Change to maintain block length 
			#print 'probe',probe_n
			#ogOnly(trial_i, probe_n)
			targetProbe(trial_i, probe_n, block, lastProbe = False)
		targetProbe(trial_i, probeInTrial-1, block, lastProbe = True) #probeInTrial is always 1 extra because starts at 1
		iti()
		resetTrial()

	## MONITOR
	elif block == 6: 
		#print 'monitor2',trial_i
		probeInTrial = df.iloc[trial_i, df.columns.get_loc('n_probes')]
		for probe_n in range(probeInTrial-1): ## not -1 because go through all probes as targetProbe
			#print 'probe', probe_n
			targetProbe(trial_i, probe_n, block, lastProbe = False)
		targetProbe(trial_i, probeInTrial-1, block, lastProbe = True)
		iti()
		resetTrial()

	## MAINTAIN & MONITOR
	elif block == 7: 
		#print 'mnm2',trial_i
		target(trial_i)
		delay()
		probeInTrial = df.iloc[trial_i, df.columns.get_loc('n_probes')]
		for probe_n in range(probeInTrial-1): ## not -1 because go through all probes as targetProbe
			#print 'probe', probe_n
			targetProbe(trial_i, probe_n, block, lastProbe = False)
		targetProbe(trial_i, probeInTrial-1, block, lastProbe = True)
		iti()
		resetTrial()

	# BASELINE
	elif block == 8: 
		#print 'baseline 2', trial_i
		probe_n = 0
		ogOnly(trial_i, probe_n)
		targetProbe(trial_i, probe_n, block, lastProbe = False)

	#else: 
	#	raise Warning('yikes, part 2')

# End of expt instruction slide
presentSlides(21)

slack_msg = 'Experiment finished'
slackMessage(block, slack_msg)

# Save output at the end
df.to_csv(full_filename)

win.close()
