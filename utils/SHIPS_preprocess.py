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
    # Mask missing values (indicated by 9999), and keep only TC cases (TYPE = 1)
    SHIPS_trim = SHIPS_in.mask(SHIPS_in == 9999)
    SHIPS_trim = SHIPS_trim.mask(SHIPS_trim['TYPE'] != 1)
    # 
    predictors_DYN = SHIPS_trim[PREDICTORS_sel]
    predictand = SHIPS_trim[['CASE','NAME','DATE_full','TIME',predictand_name]]
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
    # Mask missing values (indicated by 9999), and keep only TC cases (TYPE = 1)
    SHIPS_mask = SHIPS_in.mask(SHIPS_in == 9999)
    SHIPS_mask = SHIPS_mask.mask(SHIPS_mask['TYPE'] != 1)
    # Trim to selected predictors
    pred_sel_IR = SHIPS_mask[PREDICTORS_sel]
    
    #case_ind = pred_sel_IR['CASE'].unique().tolist()
    # Now, if IR00 (PC00) is nan, replace with IRM1 (PCM1)
    pred_sel_IR.IR00.fillna(pred_sel_IR.IRM1,inplace=True)
    pred_sel_IR.PC00.fillna(pred_sel_IR.PCM1,inplace=True)
    # If IR00 (PC00) is still NaN, replace with IRM3 (PCM3)
    pred_sel_IR.IR00.fillna(pred_sel_IR.IRM3,inplace=True)
    pred_sel_IR.PC00.fillna(pred_sel_IR.PCM3,inplace=True)
    case_ind = pred_sel_IR['CASE'].unique().tolist()
    # Select desired variables and fill for each case at all times
    pred_IR = pred_sel_IR[['CASE','NAME','DATE_full','TIME','TYPE']]
    pred_IR = pred_IR.set_index(['CASE'])
    pred_sel_IR = pred_sel_IR.set_index(['CASE','TIME'])
    for i_case in case_ind:
        # delete cases where we only have one sample
        if len(pred_IR.loc[i_case].shape) == 1:
            continue
        for i_IR in np.arange(0,len(IR00_var_names)):
            time_inds = pred_sel_IR['IR00'].loc[i_case]
            if [IR00_time_ind[i_IR]] in (time_inds.index.values):
                if pred_sel_IR['IR00'].loc[i_case].isnull().values.all() == False:
                    pred_IR.loc[i_case,IR00_var_names[i_IR]] = pred_sel_IR['IR00'].loc[i_case,IR00_time_ind[i_IR]].values*np.ones(len(pred_IR['TIME'].loc[i_case]))
                else:
                    pred_IR.loc[i_case,IR00_var_names[i_IR]] = np.nan
            else:
                pred_IR.loc[i_case,IR00_var_names[i_IR]] = np.nan
    # Same for PC variables
    for j_case in case_ind:
        # delete cases where we only have one sample
        #if j_case in (9567.0,12046.0,12047.0,12048.0,12049.0,13903.0,17008,17621,18277,20802):
        if len(pred_IR.loc[j_case].shape) == 1:
            continue
        for j_IR in np.arange(0,len(PC00_var_names)):
            time_inds = pred_sel_IR['PC00'].loc[j_case]
            if [PC00_time_ind[j_IR]] in (time_inds.index.values):
                if pred_sel_IR['PC00'].loc[j_case].isnull().values.all() == False:
                    pred_IR.loc[j_case,PC00_var_names[j_IR]] = pred_sel_IR['PC00'].loc[j_case,PC00_time_ind[j_IR]].values*np.ones(len(pred_IR['TIME'].loc[j_case]))
                else:
                    pred_IR.loc[j_case,PC00_var_names[j_IR]] = np.nan
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
        X_train_full = X_train.mean(level=['BASIN','CASE','NAME','DATE_full'])
        X_test_full = X_test.mean(level=['BASIN','CASE','NAME','DATE_full'])
        #
        y_train_o = y_train.mean(level=['BASIN','CASE','NAME','DATE_full'])
        y_test_o = y_test.mean(level=['BASIN','CASE','NAME','DATE_full'])
        
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

def load_processed_SHIPS(yr_start,yr_end,mask_TYPE,max_fore,interp_str,use_basin='ALL'):
    SHIPS_predictors = pd.DataFrame()
    fpath_load = 'DATA/processed/'
    if use_basin == 'ALL':
        BASIN = ['ATLANTIC','EAST_PACIFIC','WEST_PACIFIC','SOUTHERN_HEM']
    else:
        BASIN = [use_basin]
    #
    for i_name in BASIN:
        fname_load = fpath_load+'SHIPS_processed_{BASIN}_set_yrs_{yr_start}-{yr_end}_max_fore_hr_{max_fore}_{interp_str}_'\
        'land_mask_{mask_TYPE}.csv'.format(BASIN=i_name,yr_start=yr_start,yr_end=yr_end,
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
#
# 9.  apply_land_mask applies a mask to SHIPS cases based on distance to land.  We default by masking out cases within 100 km of land, though this can be changed. 
##
## inputs:
## SHIPS_in: dataframe with SHIPS predictors 
## TYPE: type of land mask we want to use (see preprocess_SHIPS_predictors.ipynb for details)
## max_time: time period over which we are checking DTL (default: 24 hrs) [hr]
## DTL_thresh: threshold at which we apply land mask. default is 100 km [km]
## scale: for one of the land mask type options, we scale but do not mask out cases that are less than DTL_thresh from land
##
## outputs:
##  SHIPS_mask: SHIPS predictors with land mask applied

def apply_land_mask(SHIPS_in,TYPE,max_time=24,DTL_thresh=100,scale=0.1):
    cases = SHIPS_in.dropna(how='all')['CASE'].unique().tolist()
    # TYPE 1: SIMPLE_MASK: Filter out all cases where DTL < DTL_thresh at TIME = 0 or TIME = max_time
    if TYPE == 'SIMPLE_MASK':
        SHIPS_mask = SHIPS_in.set_index(['CASE','TIME'])
        for icase in cases:
            i_SHIPS_mask = SHIPS_mask.loc[icase]
            if 0 not in i_SHIPS_mask.index:
                mask = True
            elif (max_time in i_SHIPS_mask.index) & (0 in i_SHIPS_mask.index):
                mask = (i_SHIPS_mask.loc[0]['DTL'] <= DTL_thresh) | (i_SHIPS_mask.loc[max_time]['DTL'] <= DTL_thresh)
            else:
                mask = (i_SHIPS_mask.loc[0]['DTL'] <= DTL_thresh)
            if mask == True:
                SHIPS_mask.loc[icase] = np.nan
    ##
    # TYPE 2: SIMPLE_w_INT: As with SIMPLE_MASK, but filter if DTL < DTL_thresh at any time between TIME = 0 and TIME = max_time
    elif TYPE == 'SIMPLE_w_INT':
        SHIPS_mask = SHIPS_in.set_index(['CASE','TIME'])
        for icase in cases:
            i_SHIPS_mask = SHIPS_mask.loc[icase]
            i_m_trim = i_SHIPS_mask.loc[slice(0,max_time)]
            mask = (i_m_trim['DTL'] <= DTL_thresh)
            if mask.any() == True:
                SHIPS_mask.loc[icase] = np.nan
    ##
    # TYPE 3: SCALAR_MASK: For all cases where DTL at TIME = 0 or TIME = max_time is less than some DTL_thresh, 
    # multiply the DTL by a scaling factor of 0.1 and use this DTL_scalar to reduce all SHIPS predictors accordingly. 
    # If DTL <= 0, scaling factor is 0.
    # elif TYPE == 'SCALAR_MASK':
       
    # TYPE 5: no_mask.  Do not mask out any cases regardless of if they are over land
    elif TYPE == 'no_mask':
        SHIPS_mask = SHIPS_in
    #
    return SHIPS_mask