import pingouin as pg
import pandas as pd
import numpy as np

### run repeated measure ANOVAs and post hoc t-tests

## og acc
aov_og = pg.rm_anova(dv='og_acc', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_og = pg.pairwise_ttests(dv='og_acc', within='blockType', data=all_df_averaged)

## pm acc
aov_pm = pg.rm_anova(dv='pm_acc', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_pm = pg.pairwise_ttests(dv='pm_acc', within='blockType', subject = 'subj', data=all_df_averaged)

## rt
aov_rt = pg.rm_anova(dv='meanTrial_rt', within = 'blockType', subject = 'subj', data=all_df_averaged, detailed = True)
posthoc_rt = pg.pairwise_ttests(dv='meanTrial_rt', within='blockType', subject = 'subj', data=all_df_averaged)

## pm cost
aov_pmCost = pg.rm_anova(dv='pm_cost', within = 'blockType', subject = 'subj', data=pmCost_df_averaged, detailed = True)
posthoc_pmCost = pg.pairwise_ttests(dv='pm_cost', within='blockType', subject = 'subj', data=pmCost_df_averaged)


