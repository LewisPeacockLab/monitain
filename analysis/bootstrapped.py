
# Array of all of the subjects
subj_list = all_df_byTrial.subj.unique() 



def bootstrapped(trial_df, subj_list, n_iterations, x, y, type, color):

	trial_dict = dict()
	for k, v in trial_df.groupby('subj'):
		trial_dict[k] = v

	# Create dictionary to store bootstrap results
	boot_dict = dict()

	for i in range(n_iterations):
		# Create a new random subject list
		resampled_subjList = np.random.choice(subj_list, size = len(subj_list), replace = True)

		# Create a dictionary for the new resampled subject list
		resampled_dict = dict()

		# For each subject in new subject list, add all of their data
		for subj in resampled_subjList: 
			resampled_dict[subj] = trial_dict.get(subj)

		resampled_df = pd.DataFrame()
		# Merge all of the values from the dictionary together so there is one big dataframe of all new subject data
		for key, value in trial_dict.items(): 
			if value is None: 

				print(key)
		resampled_df = pd.concat(resampled_dict.values(), ignore_index = False) 

		bootstrap_data = resampled_df.groupby(['subj']).mean()
		bootstrap_lr = pg.linear_regression(bootstrap_data[x], bootstrap_data[y])
		boot_dict[i] = bootstrap_lr

	# Create df of linear regression results - include intercept and coefficient for each bootstrap iteration
	betas = pd.DataFrame(columns = ['intercept', 'coef'])
	for key in boot_dict: 
		intercept = boot_dict.get(key).coef[0]
		coef = boot_dict.get(key).coef[1]
		betas.loc[key] = [intercept, coef]


	# Plot betas	
	sea.distplot(betas.coef, color = color) 
	plt.xlabel('Coefficient')
	plt.savefig(FIGURE_PATH + type + '_bootstrap.png', dpi = 600)
	plt.close()

	return betas;



maintain_maintain_betas = bootstrapped(maintainCost_maintainAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_maintain', 'b')
monitor_monitor_betas = bootstrapped(monitorCost_monitorAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_monitor', 'r')
mnm_mnm_betas = bootstrapped(mnmCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'mnm_mnm', 'purple')

maintain_mnm_betas = bootstrapped(maintainCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'maintain_mnm', 'b')
monitor_mnm_betas = bootstrapped(monitorCost_mnmAcc_all, subj_list, 1000, 'pm_cost', 'pm_acc', 'monitor_mnm', 'r')
