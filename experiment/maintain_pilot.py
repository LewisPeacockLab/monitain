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

### Set up Slack ###

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

N_TOTAL_TRIALS = (BASELINE_TRIALS*2) + (MAINTAIN_TRIALS*3)
N_TOTAL_TRIALS_PRACT = 5 * N_BLOCKS # 5 for each baseline, maintain1, maintain2, maintain3

BLOCK_STRUCT = [106, 20, 20, 20]

# colors
color_white = [255,255, 255] #[1,1,1]
color_black = [0,0,0] #[-1,-1,-1]
color_gray = [85,85,85] #[0,0,0]
color_cyan = [0,255,255] #[0,1,1]
color_green = [0,255,0] #[0,1,0]
color_red = [255,0,0] #[1,0,0]
color_blue = [0,0,255]

FRACTAL_SIZE = [128, 128] # default that images came at

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


### Master dataframe


##### COME BACK AND WORK ON THIS LATER



### Set up PsychPy

clock = core.Clock()

# monitor calibration
mon = monitors.Monitor('testMonitor')
mon.setWidth(SCREENS[SCRN]['width_cm'])
mon.setSizePix(SCREENS[SCRN]['pixel_dims'])

# window set up

fullscreen = False if SUBJ='s999' else True # smaller screen when debugging
win = visual.Window(
	mon=mon, 
	colorspace='rgb255',
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
	size=win.size/2),
	)

# fractal positions - top, middle, bottom
stim_fractal = visual.ImageStim(
	win=win, 
	mask='circle', 
	units='pix', 
	size=fractal_size,
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


### Utility functions

### Event functions

def baseline(): 

def maintain_1(): 

def maintain_2():

def maintain_3():

## Add below functions back if running entire experiment
#def monitor():
#def mnm():


### Experiment
block_starts = [0, 106, 126, 146]
#block_starts = [0, 106, 126, 146, 166, 186, 206, 226]
pract_starts = [106, 126, 146]

for trial_i in range(N_TOTAL_TRIALS):
	block_type = df.block[trial_i][0]
	block = df.block[trial_i][1]

	#if trial_i in block_starts:


	## BASELINE
	if block == 1: 
		baseline()

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
