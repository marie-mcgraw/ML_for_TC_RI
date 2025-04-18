import numpy as np
import sys
import os
import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix,accuracy_score,precision_score,recall_score,classification_report,brier_score_loss
from sklearn.metrics import precision_recall_curve, auc, f1_score, fbeta_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from utils.SHIPS_preprocess import SHIPS_train_test_split, calc_d24_VMAX, fore_hr_averaging, SHIPS_train_test_shuffle_CLASS
from utils.SHIPS_preprocess import load_processed_SHIPS, calculate_class_weights, get_RI_classes


###
# apply_class_label applies a label (string) to each class for plotting purposes.  We have two choices--RI only, or all classes. 
#
# Inputs: 
# X_train,X_test: training / testing features
# y_train, y_test: our predictors (classes) [array]
# is_RI_only: which classification scheme are we using--RI only or all classes? [Boolean]
#
# Outputs: y_train, y_test 
#
def apply_class_label(y_train,y_test,X_train,X_test,is_RI_only):
    if is_RI_only == False:
        y_train['I_class label'] = ['RI' if x == 2 
                               else 'I' if x == 1
                               else 'SS' if x == 0
                               else 'W' if x == -1
                               else 'RW' for x in X_train['I_class']]
        #
        y_test['I_class label'] = ['RI' if x == 2 
                               else 'I' if x == 1
                               else 'SS' if x == 0
                               else 'W' if x == -1
                               else 'RW' for x in X_test['I_class']]
    else:
        y_train['I_class label'] = ['RI' if x == 1 
                               else 'no RI' for x in X_train['I_class']]
        y_test['I_class label'] = ['RI' if x == 1 
                               else 'no RI' for x in X_test['I_class']]
    #
    return y_train,y_test
#
# 3. calc_d_VMAX calculates the 24-hour change in VMAX, starting at hour init_HR (i.e., if we start at hr 6, we calculate the change between hr 6 and hr 30 (6 + 24)
## inputs:
##   SHIPS: input SHIPS data, in dataframe format
##   init_HR: beginning of 24-hour period
##   
## outputs:
##    diff: dataframe with 24-hour changes in numerical predictors 
def calc_d24_VMAX(SHIPS,init_HR):
    nlev = SHIPS.index.nlevels
    SHIPS_t0 = SHIPS.xs(init_HR,level=nlev-1)
    SHIPS_t0['DATE_full'] = pd.to_datetime(SHIPS_t0['DATE_full'])
    SHIPS_p24 = SHIPS_t0.shift(-4)
    # Calculate 24-hr change in all numerical predictors
    pred_num = ['SHRG','D200','Z850','VMAX','VMPI','DELV','RHMD','POT','DELV -12','GOES Tb',
                's(GOES Tb)','pct < -50C','storm size','PC1','PC2','PC3','PC4']
    diff = SHIPS_p24[pred_num] - SHIPS_t0[pred_num]
    #diff = diff.dropna(how='all')
    # Drop if date difference > 1 day (so we aren't comparing two different storms)
    date_diff = SHIPS_p24['DATE_full'] - SHIPS_t0['DATE_full']
    diff = diff.where(date_diff == pd.Timedelta(1,'D')).dropna(how='all')
    diff['DATE_full'] = SHIPS_t0['DATE_full']
    return diff
#
# 4.  SHIPS_train_test_split splits SHIPS data into training and testing sets based on either year or storm.  We can't just do a typical train-test split because each CASE is not completely independent (i.e., the same storm's measurements 6 hours apart are different cases but these two cases are not independent from each other). 
## inputs:
##     SHIPS: input SHIPS data, in dataframe format
##     TEST_years: array of years that are included in test set (would use storm names if we split by storms)
##     use_years: boolean indicating whether or not to split by years or by storm number (default is to split by years)
## outputs:
##     SHIPS_train: SHIPS data identified as part of the training set
##     SHIPS_test: SHIPS data identified as part of the testing set

def SHIPS_train_test_split(SHIPS,TEST_years,use_years=True):
    if use_years == True:
        # Add an is_TRAIN flag based on the year of 
        SHIPS['is_TRAIN'] = [0 if np.isin(x,TEST_years) else 1 for x in pd.to_datetime(SHIPS['DATE_full']).dt.year]
        SHIPS_train = SHIPS.where(SHIPS['is_TRAIN']==1).dropna(how='all')
        SHIPS_test = SHIPS.where(SHIPS['is_TRAIN']==0).dropna(how='all')
    else:
        raise NameError('Will add a version that splits based on storm number later!!')
    #
    return SHIPS_train,SHIPS_test
#######
## Split SHIPS predictors into training/testing data based on years, calculate 24-hr intensity change, get intensity classification, and drop unnecessary columns that will not be our final predictors.  
#
# Dependent functions (scroll up in this file for more):
#    SHIPS_train_test_split: splits SHIPS predictors into training/testing by leaving 2 randomly-selected years for testing
#    calc_d24_VMAX: calculates 24-hr changes in all predictors (primarily intended for intensity change)
#    get_RI_classes: classifies cases based on 24-hr intensity change; classes are either RI/no-RI, or follow 5 class distinction outlined above
#    fore_hr_averaging: determine whether or not we are using hr-24 predictors, or averaging predictors over hrs 0-24
#
# Inputs:
## SHIPS_predictors: pandas dataframe including all of our SHIPS predictors of interest
## test_years: array of 2 years to hold out of training set for testing
## hrs_max: maximum hour of our prediction (default = 24)
## to_predict: name of the feature we want to predict (usually, we want to predict 24-h intensity change if we are doing a regression problem, and intensity class if we are doing a classification problem
## is_RI_only: boolean; if True, we only want to differentiate between RI and non-RI.  Otherwise, we want to identify all 5 classes.
## RI_thresh: the threshhold for the change in wind speed at which we identify RI/RW [kts]. Default is 30 kts
## to_IND: names of SHIPS_predictors to set to index
## to_DROP: names of SHIPS predictors to drop (not included in final predictor set)
## DO_AVG: Boolean that determines whether or not we are averaging predictors over hours 0-24 (True), or just using hr-24 predictors (False)
## Outputs:
## diff:  Our input Pandas dataframe, but with an additional column that includes information about the storm intensity class (['I_class'])
##

##
# calc_CM_stats calculates basic statistics from the confusion matrix of classification models (developed for multicategorical classification problems).  We are interested in the following:
#   Hits: correct predictions of event occuring
#   False alarms: event was predicted to occur but did not
#   Misses: event was not predicted to occur but did occur
#   Correct negs: event was not predicted to occur and did not occur
#   POD: probability of detection; Hits/(Hits+Misses)
#   FAR: false alarm rate; False Alarms/(False Alarms + Hits)
#   PFOD: probability of false detection; False Alarms/(False Alarms + Correct Negs)
#   Threat: threat score/CSI (critical success index); Hits/(Hits + Misses + False Alarms)
#   SR: success ratio; Hits/(Hits + False Alarms)
#   Bias: total number of positives predictions / total number of observed positive
#   CSI: critical succces
#
# Inputs: 
# label_names: names of our category labels (i.e. Rapid Intensification, Rapid Weakening, etc)
# labels: actual category labels (-2, -1, 0, 1, 2)
# cm: confusion matrix (output from sklearn ConfusionMatrix function)
#
# Outputs: cm_stats_ALL: pandas dataframe with all of the confusion matrix statistics
#
def calc_CM_stats(label_names,labels,ncats,cm):
    cm_stats_ALL = pd.DataFrame()
    for i in np.arange(0,ncats):
        cm_stats = pd.DataFrame(index={i},columns={'Category','Category Names','N_predicted','N_actual','Hits','False Alarms',
                                                         'Misses','Correct Negs','POD','FAR','PFOD','SR','Threat'})
        #
       # print(labels_name[i])
        cm_stats['Category Names'] = label_names[i]
        cm_stats['Category'] = labels[i]
        cm_stats['N_predicted'] = cm.sum(axis=0)[i]
        cm_stats['N_actual'] = cm.sum(axis=1)[i]
        cm_stats['Hits'] = np.diag(cm)[i]
        cm_stats['False Alarms'] = np.sum(cm[:,i]) - cm[i,i]
        cm_stats['Misses'] = np.sum(cm[i,:]) - cm[i,i]
        cm_negs = np.delete(cm,i,0)
        cm_negs = np.delete(cm_negs,i,1)
        cm_stats['Correct Negs'] = np.sum(cm_negs)
        cm_stats['POD'] = cm_stats['Hits']/(cm_stats['Hits']+cm_stats['Misses'])
        cm_stats['FAR'] = cm_stats['False Alarms']/(cm_stats['False Alarms']+cm_stats['Hits'])
        cm_stats['PFOD'] = cm_stats['False Alarms']/(cm_stats['False Alarms']+cm_stats['Correct Negs'])
        cm_stats['Threat'] = cm_stats['Hits']/(cm_stats['Hits']+cm_stats['Misses']+cm_stats['False Alarms'])
        cm_stats['SR'] = cm_stats['Hits']/(cm_stats['Hits']+cm_stats['False Alarms'])
        cm_stats['BIAS'] = (cm_stats['Hits']+cm_stats['False Alarms'])/(cm_stats['Hits'] + cm_stats['Misses'])
        cm_stats_ALL = cm_stats_ALL.append(cm_stats)
    #
    return cm_stats_ALL
##
#
# get_class_weights calculates the weights for our unbalanced classes when identifying RI.   
#
# Inputs: SHIPS: SHIPS predictors (dataframe)
#  n_classes:  number of classes in classification model
#  RI_thresh: threshold for identifying RI (and RW) [default is 30 kts]
#  init_hr: required to calculate d_VMAX 24 hours when we are calculating class weights BEFORE doing any analysis [default is 0]
# 
#  Outputs:
#    class_size:  dataframe that includes fraction of cases in each class (per basin), and the corresponding weights. 
def calculate_class_weights_nclass(SHIPS,n_classes,RI_thresh=30,init_hr=0):
    diff = calc_d24_VMAX(SHIPS.reset_index().set_index(['BASIN','CASE','TIME']),init_hr)
    #
    if n_classes == 5:
        diff['I_class'] = [2 if x >= RI_thresh 
                               else 1 if 10 <= x < RI_thresh
                               else 0 if abs(x) < 10
                               else -1 if -RI_thresh < x <= -10
                               else -2 for x in diff['VMAX']]
    elif n_classes == 3:
        diff['I_class'] = [1 if 10 <= x 
                               else 0 if abs(x) < 10
                               else -1 for x in diff['VMAX']]
    #
    elif n_classes == 2:
        diff['I_class'] = [1 if RI_thresh <= x 
                               else 0 for x in diff['VMAX']]
    class_size = diff.reset_index().groupby(['BASIN','I_class']).count()
    class_size_ALL = class_size.sum(level=1)
    class_size_ALL['BASIN'] = 'ALL'
    #
    class_size_full = class_size['CASE'].reset_index().append(class_size_ALL[['CASE','BASIN']].reset_index())
    class_size_pct = class_size_full.set_index(['BASIN','I_class'])
    class_size_pct['FRAC'] = class_size_pct/class_size_pct.sum(level=0)
    class_size_pct['WEIGHT'] = 1/class_size_pct['FRAC']
    return class_size_pct
#
def get_scores_class_rept(X_test,y_test,BASIN,MODEL):
    if BASIN == 'ALL':
        y_pred = MODEL.predict(X_test)
        y_true = y_test['I_class']
    else:
        y_pred = MODEL.predict(X_test.xs(BASIN))
        y_true = y_test['I_class'].xs(BASIN)
    class_report = classification_report(y_true, y_pred,output_dict=True)
    cr_df = pd.DataFrame(class_report)
    return cr_df,y_true,y_pred

###
def get_train_test_split(test_years,SHIPS,to_predict,is_RI_only,to_IND,to_DROP,DO_AVG,RI_thresh,hrs_max):
    feature_names,X_train,y_train,X_test,y_test,diff_train,diff_test = SHIPS_train_test_shuffle_CLASS(SHIPS,
                                        test_years,to_predict,is_RI_only,to_IND,to_DROP,DO_AVG,RI_thresh,hrs_max)
    # Get our class labels
    y_train,y_test = apply_class_label(y_train,y_test,X_train,X_test,is_RI_only)
    # Drop the features we don't want to use
    X_train = X_train.drop(columns={to_predict})
    X_test = X_test.drop(columns={to_predict})
    feature_names = X_train.columns
    
    return X_train, X_test, y_train, y_test, feature_names, diff_train, diff_test
    
    
###
def create_gridsearch_RF(is_standard,score,max_depth,n_estimators,max_features,min_samples_leaf,k_folds,n_repeats,use_custom_wts,wts_sel,scoring_metric):
    # Class weights
    if use_custom_wts:
        class_weight = wts_sel
    else:
        class_weight = 'balanced'
    # Create pipeline
    if is_standard:
        pipe = Pipeline([('scaler',StandardScaler()),('clf',RandomForestClassifier(
                                                                class_weight=class_weight))])
    else:
        pipe = Pipeline([('clf',RandomForestClassifier(class_weight=class_weight))])
    ## Cross-validation
    cv = RepeatedStratifiedKFold(n_splits = k_folds, n_repeats=n_repeats)
    ## Gridsearch
    params = {'clf__max_depth':max_depth,
                     'clf__criterion':score,
                     'clf__max_features':max_features,
                     'clf__n_estimators':n_estimators,
                     'clf__min_samples_leaf':min_samples_leaf}
    
    grid_LRclass = GridSearchCV(pipe,param_grid=params,cv=cv,scoring=scoring_metric)
    return grid_LRclass
###
###
def create_gridsearch_LR(is_standard,solver,penalty,C_vals,max_iter,k_folds,n_repeats,use_custom_wts,wts_sel,score,no_wts=False):
    # Class weights
    if use_custom_wts:
        class_weight = wts_sel
    elif (use_custom_wts == False) & (no_wts == False):
        class_weight = 'balanced'
    #elif (use_custom_wts == False) & (no_wts == True):
     #   class_weight = None
    # Create pipeline
    if (is_standard == True) & (no_wts == False):
        pipe = Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(solver=solver,penalty=penalty,
                                                                class_weight=class_weight))])
    elif (is_standard == False) & (no_wts == False):
        pipe = Pipeline([('clf',LogisticRegression(solver=solver,penalty=penalty,
                                                                class_weight=class_weight))])
    elif (is_standard == True) & (no_wts == True):
        pipe = Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(solver=solver,penalty=penalty,
                                                                              class_weight=None))])
    elif (is_standard == False) & (no_wts == True):
        pipe = Pipeline([('clf',LogisticRegression(solver=solver,penalty=penalty,
                                                                class_weight=None))])
    ## Cross-validation
    cv = RepeatedStratifiedKFold(n_splits = k_folds, n_repeats=n_repeats)
    ## Gridsearch
    params = {'clf__C': C_vals,
             'clf__max_iter':max_iter}
    grid_LRclass = GridSearchCV(pipe,param_grid=params,cv=cv,scoring=score)
    return grid_LRclass
###
def get_scores_best_params_LR(model,X_test,y_test,basin):
    report = pd.DataFrame()
    cr,y_true,y_pred = get_scores_class_rept(X_test,y_test,basin,model)
    cr['BASIN'] = basin
    cr['C'] = model.best_params_.get('clf__C')
    cr['Max Iter'] = model.best_params_.get('clf__max_iter')
    report = report.append(cr)
    return report, y_true, y_pred
###
def get_scores_best_params_RF(model,X_test,y_test,basin):
    report = pd.DataFrame()
    cr,y_true,y_pred = get_scores_class_rept(X_test,y_test,basin,model)
    cr['BASIN'] = basin
    cr['Max Depth'] = model.best_params_.get('clf__max_depth')
    cr['Max Features'] = model.best_params_.get('clf__max_features')
    cr['N Estimators'] = model.best_params_.get('clf__n_estimators')
    cr['Min Samples Leaf'] = model.best_params_.get('clf__min_samples_leaf')
    cr['Max Iter'] = model.best_params_.get('clf__max_iter')
    report = report.append(cr)
    return report, y_true, y_pred
###
def get_confusion_matrix_LR(model,y_true,y_pred,basin,label_names,ncats):
    cm = confusion_matrix(y_true,y_pred,labels=model.classes_)
    labels = model.classes_
    cm_stats = calc_CM_stats(label_names,labels,ncats,cm)
    cm_stats['BASIN'] = basin
    cm_stats['C'] = model.best_params_.get('clf__C')
    cm_stats['Max Iter'] = model.best_params_.get('clf__max_iter')
    return cm_stats
###
def get_confusion_matrix_RF(model,y_true,y_pred,basin,label_names,ncats):
    cm = confusion_matrix(y_true,y_pred,labels=model.classes_)
    labels = model.classes_
    cm_stats = calc_CM_stats(label_names,labels,ncats,cm)
    cm_stats['BASIN'] = basin
    cm_stats['Max Depth'] = model.best_params_.get('clf__max_depth')
    cm_stats['Max Features'] = model.best_params_.get('clf__max_features')
    cm_stats['N Estimators'] = model.best_params_.get('clf__n_estimators')
    cm_stats['Min Samples Leaf'] = model.best_params_.get('clf__min_samples_leaf')
    cm_stats['Max Iter'] = model.best_params_.get('clf__max_iter')
    return cm_stats
###
def get_feature_importances_LR(model,X_train,y_train,basin,scoring):
    fi_pred = pd.DataFrame(index=X_train.columns,columns={'mean importance','std(importance)','BASIN',
                                                         'Years Out','Fold','Max Iter','C'})
    #
    if basin == 'ALL':
        res = permutation_importance(model,X_train,y_train['I_class'],scoring=scoring)
    else:
        res = permutation_importance(model,X_train.xs(basin),y_train['I_class'].xs(basin),scoring=scoring)
    fi_pred['mean importance'] = res.importances_mean
    fi_pred['std(importance)'] = res.importances_std
    fi_pred['BASIN'] = basin
    fi_pred['C'] = model.best_params_.get('clf__C')
    fi_pred['Max Iter'] = model.best_params_.get('clf__max_iter')
    return fi_pred
###
def get_feature_importances_RF(model,X_train,y_train,basin,scoring):
    fi_pred = pd.DataFrame(index=X_train.columns,columns={'mean importance','std(importance)','BASIN',
                                                         'Years Out','Fold','Max Iter','C'})
    #
    if basin == 'ALL':
        res = permutation_importance(model,X_train,y_train['I_class'],scoring=scoring)
    else:
        res = permutation_importance(model,X_train.xs(basin),y_train['I_class'].xs(basin),scoring=scoring)
    fi_pred['mean importance'] = res.importances_mean
    fi_pred['std(importance)'] = res.importances_std
    fi_pred['BASIN'] = basin
    fi_pred['Max Depth'] = model.best_params_.get('clf__max_depth')
    fi_pred['Max Features'] = model.best_params_.get('clf__max_features')
    fi_pred['N Estimators'] = model.best_params_.get('clf__n_estimators')
    fi_pred['Min Samples Leaf'] = model.best_params_.get('clf__min_samples_leaf')
    fi_pred['Max Iter'] = model.best_params_.get('clf__max_iter')
    return fi_pred
####
def get_roc_auc(X_test,basin,model,y_test,class_sel,class_label,scoring,cut='equal'):
    # Probabilistic prediction.  Output is for classes [0,1]
    class_sel = 1
    class_label = 'RI'
    if basin == 'ALL':
        ypred_prob = model.predict_proba(X_test)
        y_test_use = y_test['I_class']
    else:
        ypred_prob = model.predict_proba(X_test.xs(basin))
        y_test_use = y_test['I_class'].xs(basin)
    # Everything for RI only
    y_scores_RI = ypred_prob[:,class_sel]
    # Get Brier Skill Score
    brier_loss = [brier_score_loss(y_test_use,[y for x in range(len(y_test_use))]) for y in np.arange(0,1.01,0.1)]
    BS_ref = np.sum(y_test_use==1)/len((y_test_use))
    #
    brier = pd.DataFrame()
    brier['BS'] = brier_loss
    brier['BASIN'] = basin
    brier['CLASS'] = class_label
    brier['Threshold'] = np.arange(0,1.01,0.1)
    brier['BS_ref'] = BS_ref
    #
    p, r, thresholds = precision_recall_curve(y_test_use,y_scores_RI)
    f1 = (2*p*r)/(p+r)
    p_vs_r = pd.DataFrame(columns={'Precision','Recall','Thresholds','Cutoff Threshold','BASIN','CLASS'})
    p_vs_r['Precision'] = p[:-1]
    p_vs_r['Recall'] = r[:-1]
    p_vs_r['Thresholds'] = thresholds
    p_vs_r['F1'] = f1[:-1]

    # 
    if scoring == 'recall_weighted':
        if cut == 'equal':
            icutoff = np.where(p > r)
            cutoff = icutoff[0][0]
        else:
            icutoff = np.where(np.round(r,1) < 1 - cut)
            cutoff = icutoff[0][0]
    elif scoring == "precision_weighted":
        if cut == 'equal':
            icutoff = np.where(r > p)
            cutoff = icutoff[0][0]
        else:
            icutoff = np.where(np.round(p,1) < 1 - cut)
            cutoff = icutoff[0][0]
    elif scoring == 'f1_weighted':
        mf = p_vs_r['F1'].max()
        imf = p_vs_r.where(p_vs_r['F1'] == mf).dropna(how='all').index
        cutoff = imf[0]
        #cutoff = f1
    pr_thresh = thresholds[cutoff]
    pr_thresh_round = np.round(pr_thresh,1)
    p_vs_r['Cutoff Threshold'] = pr_thresh
    p_vs_r['BASIN'] = basin
    p_vs_r['CLASS'] = class_label
    #
    # ROC AUC curve
    fpr, tpr, auc_thresholds = roc_curve(y_test_use, y_scores_RI)
    auc_roc_score = auc(fpr, tpr)
    recall_RI = recall_score(y_test_use,y_scores_RI >= pr_thresh_round)
    pre_RI = precision_score(y_test_use,y_scores_RI >= pr_thresh_round)
    recall_noRI = recall_score(y_test_use,y_scores_RI < pr_thresh_round)
    pre_noRI = precision_score(y_test_use,y_scores_RI < pr_thresh_round)
    #
    roc_vals = pd.DataFrame(columns={'False Positive Rate','True Positive Rate','AUC Thresholds',
                                 'AUC ROC Score','BASIN','CLASS'})
    roc_vals['False Positive Rate'] = fpr
    roc_vals['True Positive Rate'] = tpr
    roc_vals['AUC Thresholds'] = auc_thresholds
    roc_vals['AUC ROC Score'] = auc_roc_score
    roc_vals['BASIN'] = basin
    roc_vals['CLASS'] = class_label
    return ypred_prob, p_vs_r, roc_vals, brier
# Calculate the area under the performance diagram curve
def calc_AUPD(p_vs_r):
    aupd = lambda y: np.trapz(y['POD'],x=y['Success Ratio'])
    aupd_scores = p_vs_r.groupby(['Model','BASIN','Fold']).apply(aupd)
    return aupd_scores
# Get Necessary quantities for performance diagram curves
def calculate_PD_curves(p_vs_r):
    p_vs_r['POD'] = p_vs_r['Recall']
    p_vs_r['FAR'] = 1 - p_vs_r['Precision']
    p_vs_r['Success Ratio'] = 1 - p_vs_r['FAR'] 
    #
    p_vs_r['CSI'] = 1/(1 / p_vs_r['Success Ratio'] + 1 / p_vs_r['POD'] - 1)
    p_vs_r['Bias'] = p_vs_r['POD'] / p_vs_r['Success Ratio']
    # Round thresholds so plotting is nicer
    p_vs_r['Thresh Round'] = p_vs_r['Thresholds'].round(2)
    return p_vs_r