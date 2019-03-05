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
from psychopy import visual, event, core#, gui


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

# Slack 
############ NEED TO ADD 



##
#gui = gui.Dlg()
#gui.addField("Subject ID:")
#gui.addField("Initials:")
#gui.addField("Slack?")
#gui.addField("Screen size:")
#gui.show()

#subj_id = gui.data[0]
#subj_initials = gui.data[1]
#slack = gui.data[2]
#screenSize = gui.data[3]
##





####################################
########## Build dataframe #########
#### for individual participant ####
####################################

columns = ['subj_id', 'block_num', 'trial', ]

## SET UP ##

data = []
coded_data = []
responses = []


####################################
############# Psychopy #############
####################################

clock = core.Clock()

# Window set up
win = visual.Window(
	size=[1024,576], #Small size of screen for testing it out
	units="pix", 
	fullscr=False, #set to True when running for real
	color=color_gray)

# Grating set up 
grating = visual.GratingStim(
	win=win,
	units="pix", 
	size=[150,150], #size of box with grating in pixels
	mask = "circle"
	) 

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

def iti(): 
	win.color = color_gray
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

## Maintain

iti()



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

