library(lme4)
library(nlme)
library(ggplot2)

# Set directory
setwd('/Users/krh2382/monitain/analysis/output/csvs')

CSV_PATH = getwd()

# Load data
data_bytrial=read.csv("ALL_BYTRIAL.csv")
data_cost_acc = read.csv("cost_acc.csv")

# Find total number of subjects (takes excluded subjs into account)
n_subjs = length(levels(data_bytrial$subj))

# Create empty list to store linear mixed effect model resutls
lme_result = vector(mode = "list", length = n_subjs)

# Loop through subjects in dataset
for (subj_i in levels(data_bytrial$subj)) {
  data_subj = subset(data_bytrial, subj==subj_i & blockType != 'Baseline')
  lme_result[[subj_i]] <- glmer(pm_acc ~ pm_cost + blockType + (1| random_effect), data = data_subj, family=binomial(link="logit"))
}

maintain_trials = subset(data_bytrial, blockType == 'Maintain')
monitor_trials = subset(data_bytrial, blockType == 'Monitor')
mnm_trials = subset(data_bytrial, blockType == 'MnM')

lm1_maintain <- glmer(pm_acc ~ pm_cost + (1 + pm_cost | subj), data = maintain_trials, family=binomial(link="logit") )
lm2_maintain <- glmer(pm_acc ~ pm_cost + (1 | subj), data = maintain_trials, family=binomial(link="logit") )
maintain_anova = anova(lm1_maintain, lm2_maintain)

lm1_monitor <- glmer(pm_acc ~ pm_cost + (1 + pm_cost | subj), data = monitor_trials, family=binomial(link="logit") )
lm2_monitor <- glmer(pm_acc ~ pm_cost + (1 | subj), data = monitor_trials, family=binomial(link="logit") )
monitor_anova = anova(lm1_monitor, lm2_monitor)

lm1_mnm <- glmer(pm_acc ~ pm_cost + (1 + pm_cost | subj), data = mnm_trials, family=binomial(link="logit") )
lm2_mnm <- glmer(pm_acc ~ pm_cost + (1 | subj), data = mnm_trials, family=binomial(link="logit") )
mnm_anova = anova(lm1_mnm, lm2_mnm)



### MAINTAIN COST V MNM ACC
# Fit a linear model for each 1) include interaction terms so x1 + x2 + x1*x2, 2) x1+x2, 3) x1
lm1_mainCost_mnmAcc <- lm(mnm_acc ~ main_cost + mon_cost + main_cost:mon_cost, data = data_cost_acc) 
lm2_mainCost_mnmAcc <- lm(mnm_acc ~ main_cost + mon_cost, data = data_cost_acc)
lm3_mainCost_mnmAcc <- lm(mnm_acc ~ main_cost, data = data_cost_acc)

# Find AIC for each model
aic_main_mnm1 = AIC(lm1_mainCost_mnmAcc)
aic_main_mnm2 = AIC(lm2_mainCost_mnmAcc)
aic_main_mnm3 = AIC(lm3_mainCost_mnmAcc)

# Run an ANOVA between each linear model fit
main_mnm_anova_12 = anova(lm1_mainCost_mnmAcc, lm2_mainCost_mnmAcc)
main_mnm_anova_13 = anova(lm1_mainCost_mnmAcc, lm3_mainCost_mnmAcc)
main_mnm_anova_23 = anova(lm2_mainCost_mnmAcc, lm3_mainCost_mnmAcc)



### MONITOR COST V MNM ACC
# Fit a linear model for each 1) include interaction terms so x1 + x2 + x1*x2, 2) x1+x2, 3) x1
lm1_monCost_mnmAcc <- lm(mnm_acc ~ mon_cost + main_cost + mon_cost:main_cost, data = data_cost_acc) 
lm2_monCost_mnmAcc <- lm(mnm_acc ~ mon_cost + main_cost, data = data_cost_acc)
lm3_monCost_mnmAcc <- lm(mnm_acc ~ mon_cost, data = data_cost_acc)

lm4_monCost_mnmAcc <- lm(mnm_acc ~ 1, data = data_cost_acc)

# Find AIC for each model
aic_mon_mnm1 = AIC(lm1_monCost_mnmAcc)
aic_mon_mnm2 = AIC(lm2_monCost_mnmAcc)
aic_mon_mnm3 = AIC(lm3_monCost_mnmAcc)

# Run an ANOVA between each linear model fit
mon_mnm_anova_12 = anova(lm1_monCost_mnmAcc, lm2_monCost_mnmAcc)
mon_mnm_anova_13 = anova(lm1_monCost_mnmAcc, lm3_monCost_mnmAcc)
mon_mnm_anova_23 = anova(lm2_monCost_mnmAcc, lm3_monCost_mnmAcc)

data_cost_acc$noEffPred=predict(lm4_monCost_mnmAcc)
data_cost_acc$monPred=predict(lm3_monCost_mnmAcc)

# Create a function 
sum.of.squares <- function(x,y) {
  x^2 + y^2
}

## Bootstrap
n_iterations = 1000
subj_list = levels(data_cost_acc$subj)
aic_cost_mnm.data <- data.frame(
  cost_mnm_interact = rep(0,n_iterations), 
  cost_mnm_noInteract = rep(0,n_iterations),
  main_mnm = rep(0,n_iterations)
  mon_mnn = rep(0, n_iterations)
)
aic_mon_mnm.data <- data.frame(
  mon_mnm1 = rep(0,n_iterations), 
  mon_mnm2 = rep(0,n_iterations),
  mon_mnm3 = rep(0,n_iterations), 
  mon_mnm4 = rep(0,n_iterations)
  one_mnm = rep(0, n_iterations)
)


for (it in 1:n_iterations) {
  # Create list of subjects for each iteration
  sampled_subjs = sample(subj_list, size = length(subj_list), replace=T)
  for (data_i in 1:length(sampled_subjs)) {
    if (data_i == 1) {
      bootstrap = data_cost_acc[data_cost_acc$subj == sampled_subjs[data_i],]
    } else {
      bootstrap = rbind(bootstrap, data_cost_acc[data_cost_acc$subj == sampled_subjs[data_i],])
    }
  }
  
  # Maintenance cost vs MnM accuracy
  lm1_cost_mnmAcc_interact <- lm(mnm_acc ~ main_cost + mon_cost + main_cost:mon_cost, data = bootstrap) 
  lm2_cost_mnmAcc <- lm(mnm_acc ~ main_cost + mon_cost, data = bootstrap)
  lm3_mainCost_mnmAcc <- lm(mnm_acc ~ main_cost, data = bootstrap)
  lm3_monCost_mnmAcc <- lm(mnm_acc ~ mon_cost, data = bootstrap)
  lm4_monCost_mnmAcc <- lm(mnm_acc ~ 1, data = bootstrap)
  aic_cost_mnm1 = AIC(lm1_cost_mnmAcc_interact)
  aic_cost_mnm2 = AIC(lm2_cost_mnmAcc)
  aic_main_mnm3 = AIC(lm3_mainCost_mnmAcc)
  aic_mon_mnm3 = AIC(lm3_monCost_mnmAcc)
  aic_mon_mnm4 = AIC(lm4_monCost_mnmAcc)
  aic_cost_mnm.data[it, ]<- c(aic_cost_mnm1, aic_cost_mnm2, aic_main_mnm3, aic_mon_mnm3, aic_mon_mnm4)
  
  # Monitor cost vs MnM accuracy
  #lm1_monCost_mnmAcc <- lm(mnm_acc ~ mon_cost + main_cost + mon_cost:main_cost, data = bootstrap) 
  #lm2_monCost_mnmAcc <- lm(mnm_acc ~ mon_cost + main_cost, data = bootstrap)
  
  
  #aic_mon_mnm1 = AIC(lm1_monCost_mnmAcc)
  #aic_mon_mnm2 = AIC(lm2_monCost_mnmAcc)
  
  #aic_mon_mnm.data[it, ] <- c(aic_mon_mnm1, aic_mon_mnm2, aic_mon_mnm3, aic_mon_mnm4)
  
  
  
}
write.csv(aic_main_mnm.data, 'aic_main_mnm.csv')
write.csv(aic_mon_mnm.data, 'aic_mon_mnm.csv')




##### ignore stuff below for now


# Not sure what to do with below stuff yet, may not keep 
subj1 = read.csv('subj1.csv')

result <- glmer(pm_acc ~ pm_cost + blockType + (1|subj), data = data_bytrial, family=binomial(link="logit"))
summary(result)

lme_result <- glmer(pm_acc ~ pm_cost + blockType + (1|subj), data = data_bytrial, family=binomial(link="logit"))
summary(lme_result)

data_bytrial[data_bytrial$subj == 's01',]

s1_lm <- glmer(pm_acc ~ pm_cost + (1 | blockType),  data = subj1, family=binomial(link="logit"))
summary(s1_lm)

for (subj in levels(data_bytrial$subj)) {print (subj)}
