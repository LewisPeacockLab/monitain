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
	size=[150,150])

psychopy.event.waitKeys()

win.close()


##Save output 
np.savetxt() 

