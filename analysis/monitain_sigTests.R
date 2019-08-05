
library(nlme)
library(multcomp)


#################
##  load data  ##
#################

all_fname <- '~/monitain/analysis/csvs/all_bySubj.csv'
pmCost_fname <- '~/monitain/analysis/csvs/pm_cost_bySubj.csv'

baseline_fname <- '~/monitain/analysis/csvs/baseline.csv'
maintain_fname <- '~/monitain/analysis/csvs/maintain.csv'
monitor_fname <- '~/monitain/analysis/csvs/monitor.csv'
mnm_fname <- '~/monitain/analysis/csvs/mnm.csv'

baseline_PM_fname <- '~/monitain/analysis/csvs/baseline_PM.csv'
maintain_PM_fname <- '~/monitain/analysis/csvs/maintain_PM.csv'
monitor_PM_fname <- '~/monitain/analysis/csvs/monitor_PM.csv'
mnm_PM_fname <- '~/monitain/analysis/csvs/mnm_PM.csv'

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


##########################################
##  rmANOVA for block type across subjs  ##
##########################################

# load_aov <- aov(acc ~ load_type + Error(subj), data=load_df)
# summary(load_aov)

all_lme <- lme(pm_acc ~ blockType, random=~1|subj, data = all_df)
anova(all_lme)
summary(glht(all_lme,
             linfct=mcp(load_type="Tukey")),
        test=adjusted(type="bonferroni"))


##########################################
##  MCB rmANOVA across category combos  ##
##########################################

model_lme <- lme(mcb ~ clpsd_cond, random=~1|subj, data=mcb_df)
anova(model_lme)

mcb_aov <- aov(model_lme, data=mcb_df)

# t test comparing phovis phosem
phovis <- subset(mcb_df,clpsd_cond=="pho/vis",mcb,drop=TRUE)
phosem <- subset(mcb_df,clpsd_cond=="pho/sem",mcb,drop=TRUE)
semvis <- subset(mcb_df,clpsd_cond=="sem/vis",mcb,drop=TRUE) #kt added
t.test(phovis,phosem,paired=TRUE,alternative="two.sided") #PHO
t.test(phovis,semvis,paired=TRUE,alternative="two.sided") #kt added, VIS
t.test(semvis,phosem,paired=TRUE,alternative="two.sided") #kt added, SEM

# One sample t test to see if diff from 0
t.test(phovis)
t.test(phosem)
t.test(semvis)

##########################################
## ANOVA for vis, sem, pho by load type ##
##########################################

vis_lme <- lme(acc ~ load_type, random=~1|subj, data=vis_df)
anova(vis_lme)
summary(glht(vis_lme,
             linfct=mcp(load_type="Tukey")))

sem_lme <- lme(acc ~ load_type, random=~1|subj, data=sem_df)
anova(sem_lme)
summary(glht(sem_lme,
             linfct=mcp(load_type="Tukey")))

pho_lme <- lme(acc ~ load_type, random=~1|subj, data=pho_df)
anova(pho_lme)
summary(glht(pho_lme,
             linfct=mcp(load_type="Tukey")))


##########################################
# t-tests for vis, sem, pho by load type #
##########################################    

# t tests for vis 
vis_single <- subset(vis_df,load_type=="single",acc,drop=TRUE)
vis_same <- subset(vis_df,load_type=="same",acc,drop=TRUE)
vis_mixed <- subset(vis_df,load_type=="mixed",acc,drop=TRUE)
t.test(vis_single,vis_same,paired=TRUE,alternative="two.sided") 
t.test(vis_single,vis_mixed,paired=TRUE,alternative="two.sided") 
t.test(vis_same,vis_mixed,paired=TRUE,alternative="two.sided") 


# t tests for sem
sem_single <- subset(sem_df,load_type=="single",acc,drop=TRUE)
sem_same <- subset(sem_df,load_type=="same",acc,drop=TRUE)
sem_mixed <- subset(sem_df,load_type=="mixed",acc,drop=TRUE)
t.test(sem_single,sem_same,paired=TRUE,alternative="two.sided") 
t.test(sem_single,sem_mixed,paired=TRUE,alternative="two.sided") 
t.test(sem_same,sem_mixed,paired=TRUE,alternative="two.sided") 

# t tests for vis 
pho_single <- subset(pho_df,load_type=="single",acc,drop=TRUE)
pho_same <- subset(pho_df,load_type=="same",acc,drop=TRUE)
pho_mixed <- subset(pho_df,load_type=="mixed",acc,drop=TRUE)
t.test(pho_single,pho_same,paired=TRUE,alternative="two.sided") 
t.test(pho_single,pho_mixed,paired=TRUE,alternative="two.sided") 
t.test(pho_same,pho_mixed,paired=TRUE,alternative="two.sided") 



