library(lme4)
library(nlme)
library(ggplot2)

# Set directory
setwd('/Users/krh2382/monitain/analysis/output/csvs')

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

n_iterations = 1000
subj_list = 

bootstrapped <- function(subj_list, n_iterations, x, y) {
  for (subj in levels(data_cost_acc$subj)) {
    
    
    
  }
}

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
