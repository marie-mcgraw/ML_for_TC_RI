import numpy as np
import sys
import os
import pandas as pd
# 1. create_SHIPS_predictors_dyn selects desired dynamical predictors, masks missing values (demarcated by 9999s), and calculates POT. Finally, if we select is_interp = True, we'll linearly interpolate over missing values for our dynamic predictors. We specify the desired predictors in PREDICTORS_sel. 
## inputs:
##   SHIPS_in: input SHIPS data, in dataframe format
##   PREDICTORS_sel: list of predictors we want to keep
##   predictand_name: string referring to what we are trying to predict
##   is_INTERP: Boolean that indicates whether or not we will linearly interpolate over missing values
##   FORE_use: indicates which hours we are forecasting for 
##   calc_POT: Do we want to calculate potential intensity, or just keep VMPI, VMAX, and SST separate? Default is True (calculate POT)
##

##outputs: predictors_DYN_return is our masked, interpolated (if desired), and trimmed subset of the SHIPS data containing only our desired dynamical predictors
def create_SHIPS_predictors_dyn(SHIPS_in,PREDICTORS_sel,predictand_name,is_INTERP,FORE_use,calc_POT=True):
    predictors_DYN = SHIPS_in[PREDICTORS_sel]
    predictand = SHIPS_in[['ATCFID','CASE','NAME','DATE_full','TIME',predictand_name]]
    predictand = predictand.mask(predictand == 9999)
    # Calculate POT and replace all DELV with DELV at -12
    predictors_DYN_x = predictors_DYN.set_index(['CASE','TIME'])
    predictors_DYN_x = predictors_DYN_x.mask(predictors_DYN_x == 9999)
    predictors_DYN_x['DELV -12'] = np.nan
    # 
    cases = predictors_DYN_x.reset_index()['CASE'].unique().tolist()
    #
    pred_pot = pd.Series(index=predictors_DYN_x.index,name='POT')
    for i_case in cases:
    #i_case = 9002
        ipred = predictors_DYN_x.loc[i_case]
        TIME_ind = ipred.index
        if calc_POT:
            
            if 0 not in TIME_ind:
                pred_pot.loc[i_case] = np.nan
            else:
                ipot = ipred['VMPI'] - ipred['VMAX'].loc[0]
                pred_pot.loc[i_case] = ipot.values
        
        # DELV -12
        if -12 not in TIME_ind:
            predictors_DYN_x['DELV -12'].loc[i_case] = np.nan
        else:
            predictors_DYN_x['DELV -12'].loc[i_case] = ipred['DELV'].loc[-12]
    ##
    predictors_DYN_x['POT'] = pred_pot

    ## Now, we'll interp if desired. First trim to desired forecast hours
    predictors_DYN_fore = predictors_DYN_x.reset_index()
    predictors_DYN_fore = predictors_DYN_fore[predictors_DYN_fore['TIME'].isin(FORE_use)].reset_index()
    #CASE_ind = predictors_DYN_fore.reset_index()['CASE'].unique().tolist()
    predictors_DYN_fore = predictors_DYN_fore[predictors_DYN_fore['TIME'].isin(FORE_use)]
    CASE_ind = predictors_DYN_fore['CASE'].unique().tolist()
    if is_INTERP:
        print('interpolating over missing values')
        predictors_DYN_fore = predictors_DYN_fore.reset_index().set_index(['CASE','TIME'])
        predictors_DYN_interp = pd.DataFrame(index=predictors_DYN_fore.index,columns=predictors_DYN_fore.columns)
        # Loop through each case
        for icase in CASE_ind:
            no_nans = predictors_DYN_fore.loc[icase,:]['SHRG'].isna().sum()
            pred_sel = predictors_DYN_fore.loc[icase,:].reset_index()
            pred_sel['CASE'] = icase
            # Do not interpolate if more than half of the values are NaN
            if no_nans > 0.5*len(predictors_DYN_fore.loc[icase]):
                predictors_DYN_interp.loc[icase,:] = pred_sel.set_index(['CASE','TIME'])
            else:
            # If we don't have too many NaNs, interpolate. 
                i_pred = pred_sel.interpolate()
                i_pred = i_pred.set_index(['CASE','TIME'])
                predictors_DYN_interp.loc[icase,:] = i_pred
        predictors_DYN_return = predictors_DYN_interp
    else:
        predictors_DYN_return = predictors_DYN_fore
        #
    return predictors_DYN_return
#############
##2. create_SHIPS_predictors_IR. Since the IR predictors are not time-dependent, we process them differently. Each time step for the IR variables corresponds to a different IR variable at time = 0 (i.e., average GOES Ch4 brightness temperature calculated over different radii). This function masks missing data, and then fills in IR00/PC00 with IRM1/PCM1 when IR00/PC00 is not available.  If IRM1/PCM1 are also not available, we use IRM3/PCM3.  If all 3 are not available, the data remains as a NaN. We typically only want a small subset of IR predictors, indicated by IR00_time_ind/PC00_time_ind.
## inputs:
##   SHIPS_in: input SHIPS data, in dataframe format
##   PREDICTORS_sel: list of predictors we want to keep
##   FORE_use: indicates which hours we are forecasting for 
##   IR00_time_ind: the forecast times that correspond to our desired IR00 predictors
##   IR00_var_names: names for our desired IR00 predictors (to be used for output data) 
##   PC00_time_ind: the forecast times that correspond to our desired PC00 predictors
##   PC00_var_names: names for our desired PC00 predictors (to be used for output data)  
##
##outputs: predictors_IR_return is our masked, interpolated (if desired), and trimmed subset of the SHIPS data containing only our desired IR predictors

def create_SHIPS_predictors_IR(SHIPS_in,PREDICTORS_sel,FORE_use,IR00_time_ind,IR00_var_names,
                               PC00_time_ind,PC00_var_names):
    # Trim to selected predictors
    pred_sel_IR = SHIPS_in[PREDICTORS_sel]
    # mask NaNs
    pred_sel_IR = pred_sel_IR.mask(pred_sel_IR == 9999)
    case_ind = pred_sel_IR['CASE'].unique().tolist()
    # Now, if IR00 (PC00) is nan, replace with IRM1 (PCM1)
    pred_sel_IR.IR00.fillna(pred_sel_IR.IRM1,inplace=True)
    pred_sel_IR.PC00.fillna(pred_sel_IR.PCM1,inplace=True)
    # If IR00 (PC00) is still NaN, replace with IRM3 (PCM3)
    pred_sel_IR.IR00.fillna(pred_sel_IR.IRM3,inplace=True)
    pred_sel_IR.PC00.fillna(pred_sel_IR.PCM3,inplace=True)
    # Select desired variables and fill for each case at all times
    pred_IR = pred_sel_IR[['ATCFID','CASE','NAME','DATE_full','TIME','TYPE']]
    pred_IR = pred_IR.set_index(['CASE'])
    pred_sel_IR = pred_sel_IR.set_index(['CASE','TIME'])
    for i_case in case_ind:
        for i_IR in np.arange(0,len(IR00_var_names)):
            time_inds = pred_sel_IR['IR00'].loc[i_case]
            if [IR00_time_ind[i_IR]] in (time_inds.index.values):
                pred_IR.loc[i_case,IR00_var_names[i_IR]] = pred_sel_IR['IR00'].loc[i_case,IR00_time_ind[i_IR]]
            else:
                pred_IR.loc[i_case,IR00_var_names[i_IR]] = np.nan
    # Same for PC variables
    for j_case in case_ind:
        for j_IR in np.arange(0,len(PC00_var_names)):
            #
            time_inds = pred_sel_IR['PC00'].loc[j_case]
            if [PC00_time_ind[j_IR]] in (time_inds.index.values):
                pred_IR.loc[j_case,PC00_var_names[j_IR]] = pred_sel_IR['PC00'].loc[j_case,PC00_time_ind[j_IR]]
            else:
                pred_IR.loc[j_case,PC00_var_names[j_IR]] = np.nan
    # Keep only desired forecast hours
    pred_IR_return = pred_IR.reset_index()[pred_IR.reset_index()['TIME'].isin(FORE_use)].set_index(['CASE','TIME'])
    #
    return pred_IR_return
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
#
# 5. get_RI_classes identifies our classes for Rapid Intensification (RI). Rapid intesification occurs when 24-hr change in VMAX exceeds RI_thresh (default threshold is 30 kts). Rapid weakening occurs when 24-hour change in VMAX is lower than -RI_thresh.  Steady state occurs when the 24-hour change in VMAX is between +/- 10 kts.  Intensification: 24-hr change in VMAX between 10-RI_thresh kts; weakening, 24-hr change in VMAX between -10 and -RI_thresh kts. We can also choose whether or not to identify only RI cases and non-RI cases. 
## inputs:
##     diff: 24-hr change in SHIPS predictors, in dataframe format
##     RI_only: do we want to classify only RI vs non RI (True), or all 5 categories of intensification (False, default behavior)
##     RI_thresh: threshhold of 24-hr change in winds that signifies rapid intensification has taken place [in kts]. Defaults is 30 kts. 
## outputs:
##     diff: dataframe containing 24-hour change in SHIPS predictors with added flag for STORM_class

# 
def get_RI_classes(diff,RI_only=False,RI_thresh=30):
    # if RI_only = True, differentiate only between RI and everything else. Otherwise use all categories
    if RI_only == True:
        diff['STORM_class'] = [1 if x >= RI_thresh else 0 for x in diff['VMAX']]
    else:
        diff['STORM_class'] = [2 if x >= RI_thresh 
                       else -2 if x<=-RI_thresh 
                       else 1 if 10<=x<RI_thresh 
                       else -1 if -RI_thresh<=x<-10
                       else 0 for x in diff['VMAX']]
    return diff
# 
# fore_hr_averaging decides whether or not to average our predictors over our prediction period, or to keep each 6-hourly measurement separate.  We'll default to averaging over the prediction period. 
##
## inputs:
##     X_train, X_test: training and testing sets of features, respectively [dataframes]. Dataframes should be in multi-index form with the following indices: BASIN, CASE, NAME, DATE_full, TIME
##     y_train,y_test: training and testing sets of our predictands, respedtively [dataframes]. Dataframes should be in multi-index form with the following indices: BASIN, CASE, NAME, DATE_full, TIME
##     DO_AVG: True if we want to average X_train/X_test and y_train/y_test over the prediction period; False if we want to keep each 6-hrly measurement separate (i.e., if we want to differentiate between predictors at 6 and 12 hrs). Default behavior is True. If we want to keep all hours separate, we will add a suffix to the feature names for bookkeeping purposes. 
## outputs:
##     X_train_full, X_test_full: training/testing sets of features, processed according to whether or not we are averaging. [dataframes].  Output will be multi-index dataframes with indices of BASIN, CASE, NAME, DATE_full.
##     y_train_o, y_test_o: training/testing sets of predictands, processed according to whether or not we are averaging. [dataframes].  Output will be multi-index dataframes with indices of BASIN, CASE, NAME, DATE_full.


def fore_hr_averaging(X_train,X_test,y_train,y_test,DO_AVG=True):
    if DO_AVG == False:
        print('keeping all hours separate')
        X_train_0 = X_train.xs(0,level='TIME').add_suffix('_T0')
        X_train_6 = X_train.xs(6,level='TIME').add_suffix('_T6')
        X_train_12 = X_train.xs(12,level='TIME').add_suffix('_T12')
        X_train_18 = X_train.xs(18,level='TIME').add_suffix('_T18')
    ##
        X_train_full = pd.concat([X_train_0,X_train_6,X_train_12,X_train_18],axis=1)
    ##
        X_test_0 = X_test.xs(0,level='TIME').add_suffix('_T0')
        X_test_6 = X_test.xs(6,level='TIME').add_suffix('_T6')
        X_test_12 = X_test.xs(12,level='TIME').add_suffix('_T12')
        X_test_18 = X_test.xs(18,level='TIME').add_suffix('_T18')
    #
        X_test_full = pd.concat([X_test_0,X_test_6,X_test_12,X_test_18],axis=1)
        y_train_o = y_train
        y_test_o = y_test
    else:
        print('averaging hours together')
        X_train_full = X_train.mean(level=['BASIN','ATCFID','CASE','NAME','DATE_full'])
        X_test_full = X_test.mean(level=['BASIN','ATCFID','CASE','NAME','DATE_full'])
        #
        y_train_o = y_train.mean(level=['BASIN','ATCFID','CASE','NAME','DATE_full'])
        y_test_o = y_test.mean(level=['BASIN','ATCFID','CASE','NAME','DATE_full'])
        
    return X_train_full,X_test_full,y_train_o,y_test_o
#######
## Define our classes for storm intensification/RI.  We want to be able to identify either: cases that are undergoing RI versus cases that are not; OR to place all cases into one of 5 classes.
#
#  Rapid intensification: 24 hour change in VMAX that is at least RI_thresh kts; value of +2
#  Intensification: 24 hour change in VMAX that is at least 10 kts but less than RI_thresh kts; value of +1
#  Steady-state: 24 hour change in VMAX is between -10 and +10 kts; value of 0
#  Weakening: 24 hour change in VMAX that is less than -10 kts but greater than -RI_thresh kts; value of -1
#  Rapid Weakening: 24 hour change in VMAX that is less than -RI_thresh kts; value of -2
#
# Inputs:
## diff: pandas dataframe including the change in VMAX (how we determine intensification class)
## RI_only: boolean; if True, we only want to differentiate between RI and non-RI.  Otherwise, we want to identify all 5 classes.
## RI_thresh: the threshhold for the change in wind speed at which we identify RI/RW [kts]. Default is 30 kts
##
## Outputs:
## diff:  Our input Pandas dataframe, but with an additional column that includes information about the storm intensity class (['I_class'])
##
def get_RI_classes(diff,RI_only,n_classes=0,RI_thresh=30):
    # if RI_only = True, differentiate only between RI and everything else. Otherwise use all categories
    if RI_only == True:
        diff['I_class'] = [1 if x >= RI_thresh else 0 for x in diff['VMAX']]
        diff['I_class label'] = ['RI' if x >= RI_thresh else 'not RI' for x in diff['VMAX']]
    elif n_classes == 5:
        diff['I_class'] = [2 if x >= RI_thresh 
                       else -2 if x<=-RI_thresh 
                       else 1 if 10<=x<RI_thresh 
                       else -1 if -RI_thresh<x<=-10
                       else 0 for x in diff['VMAX']]
        #
        diff['I_class label'] = ['RI' if x >= RI_thresh 
                       else 'RW' if x<=-RI_thresh 
                       else 'I' if 10<=x<RI_thresh 
                       else 'W' if -RI_thresh<x<=-10
                       else 'SS' for x in diff['VMAX']]
    elif n_classes == 3:
        diff['I_class'] = [1 if 10<=x
                       else -1 if x<=-10
                       else 0 for x in diff['VMAX']]
        #
        diff['I_class label'] = ['I' if 10<=x 
                       else 'W' if x<=-10
                       else 'SS' for x in diff['VMAX']]
    return diff
## 
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
def SHIPS_train_test_shuffle_CLASS(SHIPS_predictors,test_years,to_predict,is_RI_only,to_IND,to_DROP,DO_AVG,n_classes,RI_thresh=30,hrs_max=24):
    SHIPS_train,SHIPS_test = SHIPS_train_test_split(SHIPS_predictors,test_years,True)
    # Select desired hours for predictions
    SHIPS_train = SHIPS_train[SHIPS_train['TIME'].isin(np.arange(0,hrs_max+1))]
    SHIPS_test = SHIPS_test[SHIPS_test['TIME'].isin(np.arange(0,hrs_max+1))]
    # 
    SHIPS_train = SHIPS_train.set_index(['BASIN','ATCFID','CASE','TIME'])
    SHIPS_test = SHIPS_test.set_index(['BASIN','ATCFID','CASE','TIME'])
    # Are we predicting <code>VMAX</code> or the change in <code>VMAX</code>? 
    #if to_predict == 'd_VMAX':
    diff_train = calc_d24_VMAX(SHIPS_train,0)
    diff_train = get_RI_classes(diff_train,is_RI_only,n_classes,RI_thresh)
    diff_train = diff_train.rename(columns={'VMAX':'d24_VMAX'})
    predict_train = diff_train[[to_predict]]
    diff_train = diff_train.drop(columns={to_predict})
    #
    diff_test = calc_d24_VMAX(SHIPS_test,0)
    diff_test = get_RI_classes(diff_test,is_RI_only,n_classes,RI_thresh)
    diff_test = diff_test.rename(columns={'VMAX':'d24_VMAX'})
    predict_test = diff_test[[to_predict]]
    diff_test = diff_test.drop(columns={to_predict})
    # Join and then get predictands
    SHIPS_train_all = SHIPS_train.join(predict_train).reset_index().set_index(to_IND)
    SHIPS_test_all = SHIPS_test.join(predict_test).reset_index().set_index(to_IND)
    y_train_f = SHIPS_train_all[[to_predict]].dropna(how='all')
    y_test_f = SHIPS_test_all[[to_predict]].dropna(how='all')
    # Drop redundant columns and drop hour 24 from predictors
    #SHIPS_train_all = SHIPS_train_all.reset_index().set_index(to_IND)
    SHIPS_train_d = SHIPS_train_all.drop(columns=to_DROP)
    #
    #SHIPS_test_all = SHIPS_test_all.reset_index().set_index(to_IND)
    SHIPS_test_d = SHIPS_test_all.drop(columns=to_DROP)
    X_train = SHIPS_train_d.drop([24],axis=0,level='TIME')
    X_test = SHIPS_test_d.drop([24],axis=0,level='TIME')
    # Decide on whether or not to average over all time periods
    X_train_full,X_test_full,y_train,y_test = fore_hr_averaging(X_train,X_test,y_train_f,y_test_f,DO_AVG)
    # Discard all cases for which we do not have a predictand
    X_train_trim = X_train_full.loc[y_train.index.values]
    X_test_trim = X_test_full.loc[y_test.index.values]
    X_train_trim = X_train_trim.dropna(how='any')
    X_test_trim = X_test_trim.dropna(how='any')
    y_train = y_train.loc[X_train_trim.index.values]
    y_test = y_test.loc[X_test_trim.index.values]
    #
    feature_names = X_train_trim.columns
    # Shuffle data
    fX_train_trim = X_train_trim.reindex(np.random.permutation(X_train_trim.index))
    fy_train = y_train.reindex(fX_train_trim.index)
    fX_test_trim = X_test_trim.reindex(np.random.permutation(X_test_trim.index))
    fy_test = y_test.reindex(fX_test_trim.index)
    return feature_names,fX_train_trim,fy_train,fX_test_trim,fy_test,diff_train,diff_test
##
# 7. load_processed_SHIPS loads the SHIPS predictors that have been preprocessed overall (data have been cleaned, TC cases only, scaling factors applied) and to specific requirements (how was the land mask applied, how many hours are we forecasting over, etc) 
## inputs:
##.  max_fore: maximum forecast hours [usually 24 or 48]
##   mask_TYPE: how are we handling cases close to land? [SIMPLE_MASK or no_MASK]
##.  interp_str: Did we interpolate over missing data or not? [INTERP: yes, no_INTERP: no]
##   pred_set:  Set of predictors we used.  Default is 'BASIC' but if we want to change the predictors we are using we can do that.
##.  yr_start:  First year of training data [2010 or 2005, generally]
##.  yr_end:  Last year of training data [2020; data actually ends on 12/31/2019]
##.  use_basin:  Default is to use all basins, but if we just want to use one basin, we can specify that here [ATLANTIC, EAST_PACIFIC, WEST_PACIFIC, and SOUTH_PACIFIC are the choices]
## outputs:
##     SHIPS_predictors: dataframe containing SHIPS predictors for whichever basins we care about
##.    BASIN:  list of all basins included in SHIPS_predictors

def load_processed_SHIPS(yr_start,yr_end,mask_TYPE,max_fore,interp_str,use_basin='ALL',pred_set='BASIC'):
    SHIPS_predictors = pd.DataFrame()
    fpath_load = '~/SHIPS/SHIPS_processed_'
    if use_basin == 'ALL':
        BASIN = ['ATLANTIC','EAST_PACIFIC','WEST_PACIFIC','SOUTH_PACIFIC']
    else:
        BASIN = [use_basin]
    #
    for i_name in BASIN:
        fname_load = fpath_load+'{BASIN}_{pred_set}_set_yrs_{yr_start}-{yr_end}_max_fore_hr_{max_fore}_{interp_str}_'\
        'land_mask_{mask_TYPE}.csv'.format(BASIN=i_name,pred_set=pred_set,yr_start=yr_start,yr_end=yr_end,
                                          max_fore=max_fore,interp_str=interp_str,mask_TYPE=mask_TYPE)
        iload = pd.read_csv(fname_load)
        # Change RSST / RHCN to NSST / NOHC just to keep naming consistent
        if (i_name != 'ATLANTIC') | (i_name != 'EAST_PACIFIC'):
            iload = iload.rename(columns={'RSST':'NSST','RHCN':'NOHC'})
        #
        iload['BASIN'] = i_name
        SHIPS_predictors = SHIPS_predictors.append(iload)
        #
    SHIPS_predictors = SHIPS_predictors.drop(columns={'level_0','index'})
    return SHIPS_predictors,BASIN
# 8. Calculate_class_weights calculates the weights for each class if we want to use custom class weighting.  
##
## inputs:
## SHIPS: dataframe with SHIPS predictors 
## n_classes: number of classes
## RI_thresh: threshold at which RI occurs. default is 30
## init_hr:  start hour for n-hour forecast. default is 0
##
## outputs:
##  class_size_pct: percentage of cases in each class.
def calculate_class_weights(SHIPS,n_classes,RI_thresh=30,init_hr=0):
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
# 9. apply_land_mask: If desired, mask out points within a certain distance from land. Mask can be simple or complex (will add later). 
def apply_land_mask(SHIPS,mask_type,to_IND):
    if mask_type == 'SIMPLE_MASK':
        DTL = 100
        print('applying mask')
        SHIPS.loc[:,~SHIPS.columns.isin(to_IND)] = SHIPS.loc[:,~SHIPS.columns.isin(to_IND)].mask(SHIPS['DTL']<=DTL)
    else:
        raise SyntaxError("Haven't coded this up yet")
    return SHIPS
        