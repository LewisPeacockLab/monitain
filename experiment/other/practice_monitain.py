
from itertools import product, compress
import numpy as np
import pandas as pd

[ i for i in range(10)  if i != 3]

empty = []
for i in range(10): 
	if i !=3: 
		empty.append(i)


N_MIN_PROBES = 8
N_MAX_PROBES = 15
N_PROBES_PER_BLOCK = 20

LOWER_CATCH_TRIAL = 1
UPPER_CATCH_TRIAL = 7

N_BLOCKS = 8
probe_count_list = [] 

N_BLOCKS = 8

trials = range(N_MIN_PROBES, N_MAX_PROBES+1)


N_TOTAL_TRIALS = N_PROBES_PER_BLOCK * N_BLOCKS





word_cols = ['word{:d}'.format(i+1) for i in range(N_MAX_PROBES) ]
wordCond_cols = ['word{:d}_cond'.format(i+1) for i in range(N_MAX_PROBES) ]

topTheta_cols = ['topTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]
botTheta_cols = ['botTheta{:d}'.format(i+1) for i in range(N_MAX_PROBES)]

columns = ['targTheta', #angle for memory target
	 'n_probes',  #num of probes in trial 
	 'probeTheta_loc' #where target probe is on last probe
	 ]

df_columns = columns + word_cols + wordCond_cols + topTheta_cols + botTheta_cols
df_index = range(N_TOTAL_TRIALS)
df = pd.DataFrame(columns = df_columns, index = df_index)

all_targetTheta_locs = np.repeat(['top', 'bot'], N_TOTAL_TRIALS/2)
np.random.shuffle(all_targetTheta_locs)
df['probeTheta_loc'] = all_targetTheta_locs




possible_thetas = np.linspace(0,180, 18, endpoint=False)

#possible_thetas = possible_thetas[~np.isin(possible_thetas[0,90])]

possible_thetas = np.array([10, 20, 30, 40, 50, 60, 70, 80,
	100, 110, 120, 130, 140, 150, 160, 170])



np.random.choice(possible_thetas)






def pickTheta(x): 
	return np.random.choice(possible_thetas)

df['targTheta'] = df.targTheta.apply(pickTheta)

possible_thetas = np.linspace(0,180, 18, endpoint=False)

#possible_thetas = possible_thetas[~np.isin(possible_thetas[0,90])]

possible_thetas = [10, 20, 30, 40, 50, 60, 70, 80, 
	100, 110, 120, 130, 140, 150, 160, 170]

np.random.choice(possible_thetas)





catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1)

np.random.choice(catch_range)

probes = range(N_MIN_PROBES, N_MAX_PROBES+1)
probe_range = np.repeat(trials,2)

N_CATCH_PER_BLOCK = N_PROBES_PER_BLOCK - probe_range.size


new_range = np.append(catch_range, probe_range)




for i in range(N_BLOCKS): 
	np.random.shuffle(new_range)
	probe_count_list.append(new_range)




catch_range = range(LOWER_CATCH_TRIAL, UPPER_CATCH_TRIAL+ 1)

np.random.choice(catch_range)

probes = range(N_MIN_PROBES, N_MAX_PROBES+1)
probe_range = np.repeat(trials,2)

new_range = np.append(catch_range, probe_range)




probe_count_list = []
for i in range(N_BLOCKS): 
	catch_subset = np.random.choice(catch_range, size = N_CATCH_PER_BLOCK)
	probe_set = np.append(catch_subset, probe_range)
	np.random.shuffle(probe_set)
	probe_count_list.append(probe_set)

np.ravel(probe_count_list).size #should equal 160 for hnow
df['n_probes'] = np.ravel(probe_count_list)

fake_word_list = []
fake_nonword_list = []
#fake_word_list = np.repeat(['cat'], 1000)
fake_word_list = [ 'cat{:d}'.format(i+1) for i in range(5000)]
fake_nonword_list = [ 'dawg{:d}'.format(i+1) for i in range(5000)]

fake_list = np.vstack([fake_word_list, fake_nonword_list])
np.random.shuffle(fake_list)

for i in range(N_TOTAL_TRIALS): 
	n_probes = df.loc[i, 'n_probes']
	probe_loc = df.loc[i, 'probeTheta_loc']
	memTarg = df.loc[i, 'targTheta']
	#possible_thetas_minusTarg = possible_thetas[possible_thetas!=memTarg]
	possible_thetas_minusTarg = list(compress(possible_thetas, (possible_thetas != memTarg)))

	for j in range(n_probes): 

		cond = np.random.choice(['word', 'nonword'])
		

		col_name = 'word{:d}'.format(j+1)
		col_name_cond = 'word{:d}_cond'.format(j+1)

		thetaTop_col = 'topTheta{:d}'.format(j+1)
		thetaBot_col = 'botTheta{:d}'.format(j+1)

		if j+1 != n_probes: 
			top_theta = np.random.choice(possible_thetas_minusTarg)
			bot_theta = np.random.choice(possible_thetas_minusTarg)
		elif j+1 == n_probes: 
			if probe_loc == 'top': 
				top_theta = memTarg
				bot_theta = np.random.choice(possible_thetas_minusTarg)
			elif probe_loc == 'bot': 
				top_theta = np.random.choice(possible_thetas_minusTarg)
				bot_theta = memTarg
		else: 
			raise Warning('Nooooooo')

		if cond == 'word':
			rand_word = fake_word_list.pop(0)
		elif cond == 'nonword':
			rand_word = fake_nonword_list.pop(0)
		else: 
			raise Warning('noooooooooooo')


		#insert into df
		df.loc[i, col_name] = rand_word
		df.loc[i, col_name_cond] = cond

		df.loc[i, thetaTop_col] = top_theta
		df.loc[i, thetaTop_col] = bot_theta


## check for 
for i in range(N_TOTAL_TRIALS): 
	if 

df.loc[130, 'targTheta']

df.loc[130, topTheta_cols]


