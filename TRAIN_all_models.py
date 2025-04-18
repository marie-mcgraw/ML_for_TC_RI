#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

NUM_THREADS = "1"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.SHIPS_preprocess import SHIPS_train_test_split, calc_d24_VMAX, fore_hr_averaging, SHIPS_train_test_shuffle_CLASS
from utils.SHIPS_preprocess import load_processed_SHIPS, calculate_class_weights, get_RI_classes
from utils.SHIPS_ML_model_funcs import apply_class_label, calc_CM_stats, get_scores_class_rept, get_roc_auc, get_feature_importances_RF
from utils.SHIPS_ML_model_funcs import get_confusion_matrix_RF, get_scores_best_params_RF, create_gridsearch_RF, get_train_test_split
from utils.SHIPS_ML_model_funcs import get_confusion_matrix_LR, get_scores_best_params_LR, create_gridsearch_LR, get_feature_importances_LR
from utils.SHIPS_ML_model_funcs import calc_AUPD, calculate_PD_curves
from utils.SHIPS_plotting import plot_roc_curve, plot_precision_recall_vs_threshold,add_model_results,make_performance_diagram_background
from utils.SHIPS_plotting import plot_CSI_vs_bias, plot_basic_score_basin, plot_PD_curves
#
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix,accuracy_score,precision_score,recall_score,classification_report
from sklearn.metrics import precision_recall_curve, auc, f1_score, fbeta_score
from sklearn.inspection import permutation_importance
import matplotlib.colors
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utils.SHIPS_ML_model_funcs_imblearn import create_gridsearch_RF_sampler


# ##### Ignore Annoying Warnings

# In[3]:


import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
#
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore",category=ConvergenceWarning)


# ### Model Parameters
# 
# ##### SHIPS Dataset Choice
# * `max_fore`: maximum forecast hours [usually 24 or 48]
# * `mask_TYPE`: how are we handling cases close to land? [SIMPLE_MASK or no_MASK]
# * `interp_str`: Did we interpolate over missing data or not? [INTERP: yes, no_INTERP: no]
# * `yr_start`:  First year of training data [2010 or 2005, generally]
# * `yr_end_LOAD`:  Last year of full data (to find file)[2021]
# * `yr_end_TRAIN`: Last year to use in training [2018 is default]
# * `use_basin`:  Default is to use all basins, but if we just want to use one basin, we can specify that here [ATLANTIC, EAST_PACIFIC, WEST_PACIFIC, and SOUTHERN_HEM are the choices]

# In[4]:


max_fore = 24 # maximum forecast hours
mask_TYPE = 'SIMPLE_MASK' # how are we handling the land mask?
interp_str = 'INTERP' # did we interpolate?
yr_start = 2005
yr_end_LOAD = 2021
yr_end_TRAIN = 2018
use_basin = 'ALL'


# #### SHIPS analysis choices
# * `hrs_max`: maximum forecast hours (usually 24; should be same or less than `max_fore`)
# * `RI_thresh`: change in wind speed over `hrs_max` needed for RI; default threshold is `30` kt increase in wind speed in `24` hours
# * `is_RI_only`: flag for future instances of a multi-class classification problem (should always be set to `True` for now)
# * `n_classes`: related to `is_RI_only`; how many classes are we classifying into (should be `2` for now)
# * `is_standard`: flag to indicate whether or not we want to do feature scaling with `StandardScaler` (default is `True`)
# * `DO_AVG`: flag to indicate whether or not we are averaging over our forecast period or treating each 6-hrly forecast as a separate predictor (default is `True`)
# * `drop_features`: list of features to drop before model training (usually needed for preprocessing but we don't want to train the model on them).  Commonly dropped features include:
#     * `TYPE`: storm type; should be 1 everywhere (tropical cyclones only)
#     * `VMAX`: maximum surface winds; we define our classes based entirely on `VMAX` so we don't want it in our features
#     * `DELV`: we only use `DELV -12` (change in wind speeds from -12 h to 0 h) and not the change in wind speeds relative to 0 for all hours
#     * `VMPI`: we calculated `POT` (basically `VMPI` - `VMAX_0`) so we don't need to also include `VMPI`
#     * `is_TRAIN`: just a flag we use to separate training data from validation in our bootstrapped experiments; not an actual feature to train on 
# * `to_IND`: list of quantities we want to index on for our multi-index (note that these quantities will NOT be considered features)
#     * `BASIN`: ocean basin
#     * `CASE`: case number
#     * `NAME`: name of tropical cyclone
#     * `DATE_full`: date of case (YYYY-MM-DD-HH:MM:SS).  Time stamp is for `time 0`
#     * `TIME`: forecast time.  should range from `0` to `max_fore_hrs`

# In[5]:


hrs_max = 24
# Features to drop before ML model
drop_features = {'TYPE','VMAX','DELV','VMPI','is_TRAIN'}
to_IND = ['BASIN','ATCFID','CASE','NAME','DATE_full','TIME']
RI_thresh = 30
is_RI_only = True
n_classes = 2
is_standard = True
if is_standard == True:
    stand_str = 'STANDARDIZED'
else:
    stand_str = 'noSTANDARDIZED'
DO_AVG = True


# #### ML Model Hyperparameters.  This will change based on the type of model
# #### Logistic Regression model
# *  <code>Solver</code>:  For logistic regression models, we have a few choices of solver. We will stick only with solvers that can handle multi-class classification, as we want to be able to compare different formulations of this problem. We have a few options:
#  * The default solver, <code>'lbfgs'</code>, (stands for Limited-memory Broyden-Fletcher-Goldfarb-Shanno).  Approximates second derivative with gradient evaluations; only stores last few updates, so saves memory.  Not super fast.
#  * <code>sag</code>: stands for "stochastic average gradient".  A version of gradient descent.  Fast for big datasets. 
#  * <code>saga</code>: a version of <code>sag</code> that allows for <code>L1</code> regularizaiton. 
# * <code> Penalty</code>: Are we regularizing using the L1 norm (absolute-value based) or the L2 norm (least-squares based)? For <code>sag</code> and <code>lbfgs</code>, we will use the <code>L2</code> penalty; for <code>saga</code> we will use <code>L1</code>. 
# * `C_vals`: $C$ is the model's regularization parameter.  We'll explore different values of $C$ in our hyperparameter sweep.  $C$ is the main hyperparameter that we can use to tune the model
# * `max_iter`: maximum number of iterations
# #### Random Forest Model
# * <code>score</code>:  For RF models, this measures quality of a split (called `criterion` in `sklearn`).  Options are `gini` (measures Gini impurity), `log_loss`, and `entropy` (both measure Shannon information gain). 
# * <code>max_features</code>: Maximum number of features per tree.  For classification problems should be approximately $\sqrt{N_{features}}$.  For us, $\sqrt{N_{features}} \approx 4.2$, so we use `[4,5]` as options for `max_features`
# * `n_estimators`: number of decision trees. We try `[250, 500]`. 
# * `min_samples_leaf`: min. number of samples needed to create a leaf node. We try `[2,4]`
# * `max_depth`: maximum depth of each decision tree.  We try `[5,6,8]`. Note that generally RF performance improves as `max_depth` increases, but we run the risk of overfitting + model training takes longer, so this is our compromise. 
# #### Cross-Validation
# * <code>k_folds</code>: number of folds used in our cross-validation approach.  We will use a <code>Stratified K-Means cross-validation</code> since we have imbalanced classes. Default is `10`
# * `n_repeats`: number of times we repeat k-folds cross-validation process. Default is `3`

# In[6]:


# Logistic
solver = 'lbfgs'
if (solver == 'saga'):
    penalty = 'l1'
else: 
    penalty = 'l2'
C_vals = np.logspace(-2,2,5)  #normalization factor
max_iter =[100,1000,10000] #max iterations    
# RF 
score = ['gini']
# Weights
# use_custom_wts = False
# no_wts = True
# We want to predict intensity class for each case
to_predict = 'I_class'
# Model hyperparameters
max_features = [4,5]
max_depth = [5,6,8,11]
min_samples_leaf = [10]
n_estimators = [250]
# Cross-val
k_folds = 10
n_repeats = 3
fig_format = 'png'


# ##### Load our pre-processed SHIPS files

# In[7]:


def load_processed_SHIPS(yr_start,yr_end,mask_TYPE,max_fore,interp_str,use_basin='ALL'):
    SHIPS_predictors = pd.DataFrame()
    fpath_load = 'DATA/processed/'
    if use_basin == 'ALL':
        BASIN = ['ATLANTIC','CENTRAL_PACIFIC','EAST_PACIFIC','WEST_PACIFIC','SOUTHERN_HEM']
    else:
        BASIN = [use_basin]
    #
    for i_name in BASIN:
        fname_load = fpath_load+'SHIPS_processed_{BASIN}_set_yrs_{yr_start}-{yr_end}_max_fore_hr_{max_fore}_{interp_str}_'        'land_mask_{mask_TYPE}.csv'.format(BASIN=i_name,yr_start=yr_start,yr_end=yr_end,
                                          max_fore=max_fore,interp_str=interp_str,mask_TYPE=mask_TYPE)
        iload = pd.read_csv(fname_load)
        # Change RSST / RHCN to NSST / NOHC just to keep naming consistent
        if (i_name != 'ATLANTIC') | (i_name != 'EAST_PACIFIC') | (i_name != 'CENTRAL_PACIFIC'):
            iload = iload.rename(columns={'RSST':'NSST','RHCN':'NOHC'})
        #
        iload['BASIN'] = i_name
        SHIPS_predictors = SHIPS_predictors.append(iload)
        #
    SHIPS_predictors = SHIPS_predictors.drop(columns={'level_0','index'})
    return SHIPS_predictors,BASIN


# In[8]:


SHIPS_predictors,BASIN = load_processed_SHIPS(yr_start,yr_end_LOAD,mask_TYPE,max_fore,interp_str,use_basin)
#
FULL_yrs = np.arange(yr_start,yr_end_TRAIN+1,1)
SHIPS_predictors = SHIPS_predictors[pd.to_datetime(SHIPS_predictors['DATE_full']).dt.year.isin(FULL_yrs)]


# In[9]:


SHIPS_predictors['BASIN'] = SHIPS_predictors['BASIN'].replace({'CENTRAL_PACIFIC':'EAST_PACIFIC'})


# In[ ]:





# ##### Bootstrapped model training
# First, initialize some dataframes for results

# In[10]:


predicted_y_ALL = pd.DataFrame()
roc_vals_ALL = pd.DataFrame()
brier_loss_ALL = pd.DataFrame()
p_vs_r_ALL = pd.DataFrame()
fi_pred_ALL = pd.DataFrame()
fi_pred_train_ALL = pd.DataFrame()
cm_ALL = pd.DataFrame()
report_ALL = pd.DataFrame()
# 


# Next, outline our bootstrapping experiments
# ###### Bootstrapping parameters:
# 
# * `N_samples`: number of experiments
# * `ncats`: number of categories for classification (default is 2)
# * `scoring`: scoring function for ML model (we typically used `f1_weighted` as it's better for imbalanced classes)
# * `n_valid`: number of years to use for validation
# 
# ###### Overview
# 1. Of full training period (2005-2018), we randomly select `n_valid` years to use for validation.  We use a modified leave-one-year-out approach (where instead we leave `n_valid` years out. This step is handled by the `get_train_test_split` function.  Thus we divide our SHIPS predictors as well as our target variable into training and validation samples based on year. 
# 2. We set up a hyperparameter sweep using `sklearn`'s `gridsearchCV` (contained in `create_gridsearch_RF` function).  For random forest, we explore 4 hyperparameters, identified earlier in this notebook.
# 3. After identifying best hyperparameters, we train (`model.fit()`).  We train once on cases from all ocean basins.
# 4. Once training is complete, we try to predict class of our validation years.  We predict each ocean basin separately, as well as predict all ocean basins combined. We use `get_scores_best_params_RF` to get the hyperparameters for our best model, `get_confusion_matrix_RF` to get the confusion matrix and contingency table stats for our model, `get_feature_importances_RF` to get the feature importances, and `get_roc_AUC` to get the receiver operator curve (ROC) and area under the curve (AUC). 
# 5. We save all of the output and repeat the process, selecting new validation years and fully re-training every time until we have done `N_samples` experiments. 

# In[11]:


def evaluate_model_RF(model,X_test,y_test,basin,fold,model_name,test_years,label_names,ncats,scoring):
    # Classification report
    report, y_true, y_pred = get_scores_best_params_RF(model,X_test,y_test,basin)
    report['Years Out'] = str(test_years)
    report['Model'] = model_name
    report['Fold'] = i
    # Confusion matrix
    cm_stats = get_confusion_matrix_RF(model,y_true,y_pred,basin,label_names,ncats)
    cm_stats['Years Out'] = str(test_years)
    cm_stats['Model'] = model_name
    cm_stats['Fold'] = i
    # Feature importances
    fi_pred = get_feature_importances_RF(model,X_test,y_test,basin,scoring)
    fi_pred['Years Out'] = str(test_years)
    fi_pred['Model'] = model_name
    fi_pred['Fold'] = i
    # ROC curve / AUC scores
    ypred_prob, p_vs_r, roc_vals, brier = get_roc_auc(X_test,basin,model,y_test,1,'R1',scoring,'equal')
    p_vs_r['Years Out'] = str(test_years)
    p_vs_r['Model'] = model_name
    p_vs_r['Fold'] = i
    roc_vals['Fold'] = i
    roc_vals['Model'] = model_name
    roc_vals['Years Out'] = str(test_years)
    brier['Years Out'] = str(test_years)
    brier['Model'] = model_name
    brier['Fold'] = i
    # Get actual predictions of target variable Y
    if basin != 'ALL':
        y_pred_all = y_test.xs(basin).copy()
    else:
        y_pred_all = y_test.copy()
    # Save predicted values of y
    y_pred_all['Y pred'] = y_pred
    y_pred_all['Predicted Basin'] = basin
    y_pred_all['Model'] = model_name
    # Get probabilities for 0 (not-RI) and 1 (RI)
    y_pred_all['Y pred probab (class: 0)'] = ypred_prob[:,0]
    y_pred_all['Y pred probab (class: 1)'] = ypred_prob[:,1]
    #
    return y_pred_all,roc_vals,brier,p_vs_r,fi_pred,cm_stats,report


# In[12]:


def evaluate_model_LR(model,X_test,y_test,basin,fold,model_name,test_years,label_names,ncats,scoring):
    # Classification report
    report, y_true, y_pred = get_scores_best_params_LR(model,X_test,y_test,basin)
    report['Years Out'] = str(test_years)
    report['Model'] = model_name
    report['Fold'] = i
    # Confusion matrix
    cm_stats = get_confusion_matrix_LR(model,y_true,y_pred,basin,label_names,ncats)
    cm_stats['Years Out'] = str(test_years)
    cm_stats['Model'] = model_name
    cm_stats['Fold'] = i
    # Feature importances
    fi_pred = get_feature_importances_LR(model,X_test,y_test,basin,scoring)
    fi_pred['Years Out'] = str(test_years)
    fi_pred['Model'] = model_name
    fi_pred['Fold'] = i
    # ROC curve / AUC scores
    ypred_prob, p_vs_r, roc_vals, brier = get_roc_auc(X_test,basin,model,y_test,1,'R1',scoring,'equal')
    p_vs_r['Years Out'] = str(test_years)
    p_vs_r['Model'] = model_name
    p_vs_r['Fold'] = i
    roc_vals['Fold'] = i
    roc_vals['Model'] = model_name
    roc_vals['Years Out'] = str(test_years)
    brier['Years Out'] = str(test_years)
    brier['Model'] = model_name
    brier['Fold'] = i
    # Get actual predictions of target variable Y
    if basin != 'ALL':
        y_pred_all = y_test.xs(basin).copy()
    else:
        y_pred_all = y_test.copy()
    # Save predicted values of y
    y_pred_all['Y pred'] = y_pred
    y_pred_all['Predicted Basin'] = basin
    y_pred_all['Model'] = model_name
    # Get probabilities for 0 (not-RI) and 1 (RI)
    y_pred_all['Y pred probab (class: 0)'] = ypred_prob[:,0]
    y_pred_all['Y pred probab (class: 1)'] = ypred_prob[:,1]
    #
    return y_pred_all,roc_vals,brier,p_vs_r,fi_pred,cm_stats,report


# In[13]:


# Experiment parameters
N_samples = 25
ncats = 2
scoring = 'f1_weighted'
cut = 'equal'
sample_frac = 0.7
sampler = SMOTE(sampling_strategy = sample_frac)
sampler_str = 'over'
sampler2 = [RandomOverSampler(sampling_strategy = sample_frac)]
sampler_str2 = ['over']
sampler_str_ALL = [sampler_str]#,sampler_str2]
sampler_ALL = [sampler]#,sampler2]
#
# FULL_yrs = np.arange(yr_start,yr_end_TRAIN,1)
use_custom_wts = False
wts_sel = 0
n_valid = 3
label_names = ['not RI','RI']
BASIN_all = ['ATLANTIC', 'EAST_PACIFIC', 'WEST_PACIFIC', 'SOUTHERN_HEM','ALL']
# Loop through bootstrapping examples
for i in np.arange(0,N_samples):
    #i = 2
    print('running sample ',i)
    # Split data into training/validation
    test_years = np.random.choice(FULL_yrs,n_valid,replace=False) # years we will use for validation
    X_train, X_test, y_train, y_test, feature_names, diff_train, diff_test = get_train_test_split(test_years,SHIPS_predictors,to_predict,
                                                                    is_RI_only,to_IND,drop_features,DO_AVG,RI_thresh,hrs_max)
    # Set up hyperparameter sweeps
    RF_model_ov = create_gridsearch_RF_sampler(is_standard,score,max_depth,n_estimators,max_features,min_samples_leaf,
                                k_folds,n_repeats,scoring,sampler_ALL,sampler_str_ALL)
    #
    RF_model_ROS = create_gridsearch_RF_sampler(is_standard,score,max_depth,n_estimators,max_features,min_samples_leaf,
                                k_folds,n_repeats,scoring,sampler2,sampler_str2)
    #
    RF_model = create_gridsearch_RF(is_standard,score,max_depth,n_estimators,max_features,min_samples_leaf,
                                k_folds,n_repeats,use_custom_wts,wts_sel,scoring)
    # 
    LR_model = create_gridsearch_LR(is_standard,solver,penalty,C_vals,max_iter,k_folds,n_repeats,
                                    use_custom_wts,wts_sel,scoring,no_wts=False)
    # Fit models, train on cases from all 4 basins
    print('fitting models')
    LR_model.fit(X_train,y_train['I_class'])
    RF_model.fit(X_train,y_train['I_class'])
    RF_model_ov.fit(X_train,y_train['I_class'])
    RF_model_ROS.fit(X_train,y_train['I_class'])
    # Now get scores for each basin
    print('validating models')
    for basin in BASIN_all:
        print('running ',basin)
        y_pred_all_RF_OV,roc_vals_RF_OV,brier_loss_RF_OV,p_vs_r_RF_OV,fi_pred_RF_OV,cm_stats_RF_OV,report_RF_OV = evaluate_model_RF(RF_model_ov,
                              X_test,y_test,basin,i,'Random Forest (SMOTE)',test_years,label_names,ncats,scoring)
        # 
        y_pred_all_ROS,roc_vals_ROS,brier_loss_ROS,p_vs_r_ROS,fi_pred_ROS,cm_stats_ROS,report_ROS = evaluate_model_RF(RF_model_ROS,
                              X_test,y_test,basin,i,'Random Forest (random oversample)',test_years,label_names,ncats,scoring)
        #
        y_pred_all_RF,roc_vals_RF,brier_loss_RF,p_vs_r_RF,fi_pred_RF,cm_stats_RF,report_RF = evaluate_model_RF(RF_model,
                              X_test,y_test,basin,i,'Random Forest (class wt)',test_years,label_names,ncats,scoring)
        # 
        y_pred_all_LR,roc_vals_LR,brier_loss_LR,p_vs_r_LR,fi_pred_LR,cm_stats_LR,report_LR = evaluate_model_LR(LR_model,
                              X_test,y_test,basin,i,'Logistic Reg.',test_years,label_names,ncats,scoring)
        # 
        y_pred_all = pd.concat([y_pred_all_RF_OV,y_pred_all_ROS,y_pred_all_RF,y_pred_all_LR])
        predicted_y_ALL = predicted_y_ALL.append(y_pred_all)
        # 
        roc_vals = pd.concat([roc_vals_RF,roc_vals_ROS,roc_vals_RF_OV,roc_vals_LR])
        roc_vals_ALL = roc_vals_ALL.append(roc_vals)
        #
        brier_loss_vals = pd.concat([brier_loss_RF,brier_loss_ROS,brier_loss_RF_OV,brier_loss_LR])
        brier_loss_ALL = brier_loss_ALL.append(brier_loss_vals)
        #
        pvr = pd.concat([p_vs_r_RF_OV,p_vs_r_ROS,p_vs_r_RF,p_vs_r_LR])
        p_vs_r_ALL = p_vs_r_ALL.append(pvr)
        # 
        fi_pred = pd.concat([fi_pred_RF_OV,fi_pred_ROS,fi_pred_RF,fi_pred_LR])
        fi_pred_ALL = fi_pred_ALL.append(fi_pred)
        #
        cm_stats = pd.concat([cm_stats_RF_OV,cm_stats_ROS,cm_stats_RF,cm_stats_LR])
        cm_ALL = cm_ALL.append(cm_stats)
        #
        report = pd.concat([report_RF_OV,report_ROS,report_RF,report_LR])
        report_ALL = report_ALL.append(report)


# In[ ]:


# X_test
#y_test


# In[ ]:


predicted_y_ALL['BASIN'] = predicted_y_ALL['Predicted Basin'] # For naming purposes
foo = report_ALL.reset_index().rename(columns={'index':'Score'})
foo2 = foo.set_index(['Score'])
fig2,ax2 = plt.subplots(1,1,figsize=(10,6))
f2a = sns.stripplot(data=foo2.xs('recall'),x='BASIN',y='1.0',hue='Model',palette=sns.color_palette('tab10'),s=15,
              label='_nolegend_',ax=ax2)
f2b = sns.stripplot(data=foo2.xs('f1-score'),x='BASIN',y='1.0',hue='Model',palette=sns.color_palette('tab10'),s=30,
              alpha=0.5,label='_nolegend_',ax=ax2)
ax2.legend([],[],frameon=False)
f2c = sns.stripplot(data=foo2.xs('precision'),x='BASIN',y='1.0',hue='Model',palette=sns.color_palette('tab10'),s=5,
              alpha=0.5,label='_nolegend_',ax=ax2)


# ##### Save everything

# In[ ]:


save_dir = 'DATA/ML_model_results/TRAINING/'
model_type = 'all_models_ROS_and_SMOTE_sample_frac_{frac}'.format(frac=sample_frac)
save_dir = save_dir+model_type+'/'
save_extension = 'TRAIN_SHIPS_vs_no_RI_{yr_start}-{yr_end}_{mask_TYPE}_{stand_str}_RI_thresh_{RI_thresh}''_{N}_samples_{scoring}.csv'.format(yr_start=yr_start,yr_end=yr_end_TRAIN,mask_TYPE=mask_TYPE,
                           stand_str=stand_str,RI_thresh=RI_thresh,N=N_samples,scoring=scoring)
save_ext_figs = 'TRAIN_SHIPS_SIMPLE_RI_vs_no_RI_{yr_start}-{yr_end}_{mask_TYPE}_{stand_str}_RI_thresh_{RI_thresh}''_{N}_samples_{scoring}.png'.format(yr_start=yr_start,yr_end=yr_end_TRAIN,mask_TYPE=mask_TYPE,
                           stand_str=stand_str,RI_thresh=RI_thresh,N=N_samples,scoring=scoring)


# ##### Create subdirectories if they don't exist

# In[ ]:


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# figs directory
if not os.path.exists(save_dir+'/figs/'):
    os.makedirs(save_dir+'/figs/')


# In[ ]:


predicted_y_ALL.to_csv(save_dir+'PREDICTED_Y_vals'+save_extension)
print('saved y vals')
roc_vals_ALL.to_csv(save_dir+'ROC_AUC_vals'+save_extension)
print('saved ROC vals')
p_vs_r_ALL.to_csv(save_dir+'Prec_vs_recall'+save_extension)
print('saved precision / recall values')
fi_pred_ALL.to_csv(save_dir+'Feat_Imp_validation'+save_extension)
print('saved feat importances')
fi_pred_train_ALL.to_csv(save_dir+'Feat_Imp_TRAIN'+save_extension)
print('saved feat importances (training)')
cm_ALL.to_csv(save_dir+'Conf_Matrix'+save_extension)
print('saved confusion matrix')
report_ALL.to_csv(save_dir+'Class_Report'+save_extension)
print('saved classification report ',save_dir+'Class_Report'+save_extension)


# In[ ]:




