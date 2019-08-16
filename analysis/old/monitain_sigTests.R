
library(nlme)
library(multcomp)


#################
##  load data  ##
#################

all_fname <- '~/monitain/analysis/output/csvs/all_bySubj.csv'
pmCost_fname <- '~/monitain/analysis/output/csvs/pm_cost_bySubj.csv'

baseline_fname <- '~/monitain/analysis/output/csvs/baseline.csv'
maintain_fname <- '~/monitain/analysis/output/csvs/maintain.csv'
monitor_fname <- '~/monitain/analysis/output/csvs/monitor.csv'
mnm_fname <- '~/monitain/analysis/output/csvs/mnm.csv'

baseline_PM_fname <- '~/monitain/analysis/output/csvs/baseline_PM.csv'
maintain_PM_fname <- '~/monitain/analysis/output/csvs/maintain_PM.csv'
monitor_PM_fname <- '~/monitain/analysis/output/csvs/monitor_PM.csv'
mnm_PM_fname <- '~/monitain/analysis/output/csvs/mnm_PM.csv'

all_df <- read.csv(all_fname)
pmCost_df <- read.csv(pmCost_fname)

baseline_df <- read.csv(baseline_fname)
maintain_df <- read.csv(maintain_fname)
monitor_df <- read.csv(monitor_fname)
mnm_df <- read.csv(mnm_fname)

baseline_PM_df <- read.csv(baseline_PM_fname)
maintain_PM_df <- read.csv(maintain_PM_fname)
monitor_PM_df <- read.csv(monitor_PM_fname)
mnm_PM_df <- read.csv(mnm_PM_fname)


##########################################################
##  rmANOVA for PM accuracy by block type across subjs  ##
##########################################################

# load_aov <- aov(acc ~ load_type + Error(subj), data=load_df)
# summary(load_aov)
na.exclude(all_df)
pm_acc_lme <- lme(pm_acc ~ blockType, random=~1|subj, data = na.exclude(all_df)) #drop out rows with missing values
anova(pm_acc_lme)
summary(glht(pm_acc_lme,
             linfct=mcp(blockType="Tukey")),
        test=adjusted(type="bonferroni"))

##########################################################
##  rmANOVA for OG accuracy by block type across subjs  ##
##########################################################

og_acc_lme <- lme(og_acc ~ blockType, random=~1|subj, data = na.exclude(all_df)) #drop out rows with missing values
anova(og_acc_lme)
summary(glht(og_acc_lme,
             linfct=mcp(blockType="Tukey")),
        test=adjusted(type="bonferroni"))

##########################################################
##  rmANOVA for RTs by block type across subjs  ##
##########################################################

rt_lme <- lme(meanTrial_rt ~ blockType, random=~1|subj, data = na.exclude(all_df)) #drop out rows with missing values
anova(rt_lme)
summary(glht(rt_lme,
             linfct=mcp(blockType="Tukey")),
        test=adjusted(type="bonferroni"))


##########################################################
##  rmANOVA for RTs by block type across subjs  ##
##########################################################

pmCost_lme <- lme(pm_cost ~ blockType, random=~1|subj, data = na.exclude(pmCost_df)) #drop out rows with missing values
anova(pmCost_lme)
summary(glht(pmCost_lme,
             linfct=mcp(blockType="Tukey")),
        test=adjusted(type="bonferroni"))

##########################################
# t-tests by block type #
##########################################    

all_baseline <- subset(all_df, blockType=="Baseline", pm_acc, drop=TRUE)
all_maintain <- subset(all_df, blockType=="Maintain", pm_acc, drop=TRUE)
all_monitor <- subset(all_df, blockType=="Monitor", pm_acc, drop=TRUE)
all_mnm <- subset(all_df, blockType=="MnM", pm_acc, drop=TRUE)

#t.test(all_baseline, all_maintain, paired=TRUE, alternative="two.sided")
#t.test(all_baseline, all_monitor, paired=TRUE, alternative="two.sided")
#t.test(all_baseline, all_mnm, paired=TRUE, alternative="two.sided")

t.test(all_maintain, all_monitor, paired=TRUE, alternative="two.sided")
t.test(all_maintain, all_mnm, paired=TRUE, alternative="two.sided")

t.test(all_monitor, all_mnm, paired=TRUE, alternative="two.sided")

