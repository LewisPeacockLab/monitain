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
	text="word", 
	color=color_black, 
	height = 40.0)
text.draw()


win.flip()

sec = 2 
core.wait(sec)

# End loop

###Target probe


win.close() #Close window













text = psychopy.visual.TextStim(
	win=win, 
	text="Welcome to the experiment!", 
	color=color_white)
text.draw()

text.color = [-1,0,-1]
text.pos = [1,1]
text.draw()

win.flip()

grating = psychopy.visual.GratingStim(
	win=win,
	units="pix",
	size=[80,80]) #size of box with grating in pixels



grating.sf = 5.0 / 80.0 #spatialFrequency = # of cycles / length in pixels 

grating.mask = "circle" 

orientations = [0.0, 45.0, 90.0, 135.0]
grating_hpos = [-150, -50, 50, 150]

for i_grating in range(4): 
	grating.ori = orientations[i_grating]
	grating.pos = [grating_hpos[i_grating], 0]

	grating.draw()

win.flip()

psychopy.event.waitKeys()

win.close()


##Save output 
np.savetxt() 

