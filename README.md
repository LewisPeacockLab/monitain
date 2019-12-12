# monitain

## What is Monitain?

Monitain is an active behavioral experiment in the Lewis-Peacock lab that is trying to tease apart the **maintenance and monitoring components of prospective memory**. Prospective memory, memory for future intentions, is crucial for success in day to day life. According to the multiprocess theory of prospective memory (PM), there are two strategies that can be employed: proactive and reactive control (McDaniel & Einstein, 2000). Proactive control requires actively thinking about the target item you are trying to remember while proactive control involves not thinking about it until the right time. 

An issue that has been raised in the past, specifically when looking at proactive control, is what maintenance is contributing to performance and what is monitoring contributing. Can they be isolated or is one dependent on the other? This was a question posed following an earlier experiment in the lab, so this experiment was created to try to study the two components (Koslov et al., 2019). Up until this point in the literature, maintenance and monitoring have not necessarily been thought as individual, dissociable processes. They are both understood to be part of the proactive part of prospective memory but the roles they play are not well understood.


## Experiment

Monitain contains a **non-focal PM task**. In non-focal PM tasks, participants must switch between the ongoing task and monitoring for the PM target. Because we really wanted to look at monitoring, we wanted to make sure that we made participants monitor as much as possible in the monitoring block and then make sure the task was similar in the other blocks in order to compare the blocks with one another. 

The scripts to run the experiment using PsychoPy and the necessary supplemental scripts (instructions, stimuli, etc.) are in the *Experiment* folder. 


## Analysis

In behavioral experiments, PM performance is usually analyzed using a behavioral metric called **PM cost**. PM cost is the reaction time on a baseline task where you are just performing an ongoing task subtracted from the reaction time for the same ongoing task except the participant is holding onto a PM memory target. This difference in reaction time can inform us what the 'cost' to reaction time is to hold onto that memory item. Here we used PM cost in our analysis. 

PM cost is useful for predicting errors. Because of this we wanted to tease apart PM cost for just maintaining and just monitoring before looking at them together. Following this train of thought, we were able to see if maintenance or monitoring on their own were useful for predicting performance or if it was the combination of the two that was necessary for prediction. 

The Python and R scripts used to run analysis and create figures are in the *Analysis* folder. 


