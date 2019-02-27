##################################
##################################
######### monitain v1.0 ##########
######## Katie Hedgpeth ##########
######## February 2019 ###########
##################################
##################################


import random
import numpy as np 
from psychopy import visual, event, core

color_white = [1,1,1]
color_black = [-1,-1,-1]
color_gray = [0,0,0]

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

sec = 3
core.wait(sec)

win.color=color_gray
win.flip()

###OG task

# Start a loop for this

text = visual.TextStim(
	win=win, 
	text="apple", 
	color=color_black, 
	height = 40.0)
text.draw()

win.flip()

sec = 2 
core.wait(sec)

# Second word just for show

text.text = "boat"
text.draw()
win.flip()
core.wait(sec)

# End loop

###Target probe

grating_hpos = [-150, 150]

for i_grating in range(2): 
	grating.pos = [0, grating_hpos[i_grating]]
	grating.draw()

text.text = "glorb"
text.draw()

win.flip()
core.wait(sec)

win.close() #Close window

##############


## MONITOR ##

#############


## MAINTAIN AND MONITOR ##
##########################

##Save output 
np.savetxt() 

