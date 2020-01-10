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

# get subject num from entry at command line
# default is s999
parser = argparse.ArgumentParser(description='Monitain experimental display')
parser.add_argument('--subj', default='s999', type=str, help='sXXX format')
args = parser.parse_args()

SUBJ = args.subj

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
		print 'Slack messaging failed - no internet connection'

	print msg


### Directories and import data ## 

DATA_PATH = 'monitain_v1.1_MAINTAIN_' + str(SUBJ)
DATA_FNAME = DATA_PATH + '.csv'
if os.path.exists(DATA_FNAME):
	sys.exit('Filename ' + DATA_FNAME + " already exists!")

# import images
images = glob.glob('stimuli/grayscale/*.png')

word_file = 'words.csv'
nonword_file = 'nonwords.csv'
pract_word_file = 'pract_words.csv'
pract_nonword_file = 'pract_nonwords.csv'

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
N_BLOCKS_MAINTAIN_1 = 1
N_BLOCKS_MAINTAIN_1 = 1

N_TRIALS = range(N_MIN_PROBES, N_MAX_PROBES+1)

BASELINE_TRIALS = 106
MAINTAIN_TRIALS = 20

N_TOTAL_TRIALS = (BASELINE_TRIALS*2) + (MAINTAIN_TRIALS*3)
N_TOTAL_TRIALS_PRACT = 5 * N_BLOCKS # 5 for each baseline, maintain1, maintain2, maintain3

# colors
color_white = [255,255, 255] #[1,1,1]
color_black = [0,0,0] #[-1,-1,-1]
color_gray = [85,85,85] #[0,0,0]
color_cyan = [0,255,255] #[0,1,1]
color_green = [0,255,0] #[0,1,0]
color_red = [255,0,0] #[1,0,0]
color_blue = [0,0,255]

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


