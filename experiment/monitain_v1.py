##################################
##################################
######### monitain v1.0 ##########
######## Katie Hedgpeth ##########
######## February 2019 ###########
##################################
##################################


import random
import numpy as np 
import psychopy.visual
import psychopy.event

win = psychopy.visual.Window(
	size=[400,400],
	units="pix", 
	fullscr=False, #set to True when running for real
	color=[1,1,1])

text = psychopy.visual.TextStim(
	win=win, 
	text="Welcome to the experiment!", 
	color=[-1,-1,-1])
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

