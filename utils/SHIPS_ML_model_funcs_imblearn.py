import numpy as np
import sys
import os
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 
###
def create_gridsearch_LR_sampler(is_standard,solver,penalty,C_vals,max_iter,k_folds,n_repeats,score,sampler,
                                sampler_str):

    # Create pipeline
    sample_steps = [(sampler_str[i],sampler[i]) for i in np.arange(0,len(sampler))]
    if is_standard:
        if len(sample_steps) > 1:
            pipe = Pipeline([('scaler',StandardScaler()),
                         sample_steps[0],sample_steps[1],
                         ('clf',LogisticRegression(solver=solver,penalty=penalty))])
        elif len(sample_steps) == 1:
            pipe = Pipeline([('scaler',StandardScaler()),
                         sample_steps[0],
                         ('clf',LogisticRegression(solver=solver,penalty=penalty))])
    else:
        if len(sample_steps) > 1: 
            pipe = Pipeline([sample_steps[0],sample_steps[1],
                             ('clf',LogisticRegression(solver=solver,penalty=penalty))])
        elif len(sample_steps) == 1:
            pipe = Pipeline([(sampler_str,sampler),
            ('clf',LogisticRegression(solver=solver,penalty=penalty))])
    ## Cross-validation
    cv = RepeatedStratifiedKFold(n_splits = k_folds, n_repeats=n_repeats)
    ## Gridsearch
    params = {'clf__C': C_vals,
             'clf__max_iter':max_iter}
    grid_LRclass = GridSearchCV(pipe,param_grid=params,cv=cv,scoring=score)
    return grid_LRclass