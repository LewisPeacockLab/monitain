library(lme4)
library(nlme)
library(ggplot2)

# Set directory
setwd('/Users/krh2382/monitain/analysis/output/csvs')

# Load data
data_bytrial=read.csv("ALL_BYTRIAL.csv")

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

lm1_monitor <- glmer(pm_acc ~ pm_cost + (1 + pm_cost | subj), data = monitor_trials, family=binomial(link="logit") )
lm2_monitor <- glmer(pm_acc ~ pm_cost + (1 | subj), data = monitor_trials, family=binomial(link="logit") )

lm1_mnm <- glmer(pm_acc ~ pm_cost + (1 + pm_cost | subj), data = mnm_trials, family=binomial(link="logit") )
lm2_mnm <- glmer(pm_acc ~ pm_cost + (1 | subj), data = mnm_trials, family=binomial(link="logit") )

# Run ANOVA on 2 models

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
