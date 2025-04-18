#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, glob, datetime, warnings, sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from utils.RT_SHIPS_preprocess import apply_land_mask, create_SHIPS_predictors_IR, create_SHIPS_predictors_dyn


# ### `get_all_lsdiag(ex_dir,ex_file)` <a name="get_all_lsdiag"></a>
# This function gets a list of all available `lsdiag` files for a given forecast time in a given basin. The `lsdiag` file naming convention is as follows: `YYMMDDHHBBNNYY_lsdiag.dat`, where `BB` refers to the ocean basin (`AL` for Atlantic, `EP` for East Pacific), and `NN` refers to the storm number. The user will provide the first part of the filename: `YYMMDDHHBB`, as there may be multiple storms for a given forecast time. 
# 
# <b>Inputs:</b>
# * `ex_dir`: directory where `lsdiag` files are located [string]
# * `ex_file`: naming prefix for `lsidag` files; should be `YYMMDDHHBB` [string]
# 
# <b>Outputs:</b>
# * `files`: list of all `lsdiag` files for given forecast time in specified basin [list of strings]

# In[2]:


def get_all_lsdiag(ex_dir,ex_file):
    # 1. Get all lsdiag files for given day in given basin
    files = [os.path.join(ex_dir, f) for f in os.listdir(ex_dir) if ((f.startswith(ex_file)) & (f.endswith('_lsdiag.dat')))]
    return files
    


# ### `read_in_lsdiag(fname)` <a name="read_in_lsdiag"></a>
# This function reads in the data from a given `lsdiag` file. Due to formatting conventions, we read the file in line by line. Note that if this function crashes or returns an empty dataframe, line length should be the first thing to check. 
# <b>Inputs:</b>
# * `fname`: name of `lsdiag` file to read [string]
# 
# <b>Outputs:</b>
# * `df_new`: dataframe containing contents of `lsdiag` [Pandas dataframe]

# In[3]:


def read_in_lsdiag(fname):
    df_new = pd.DataFrame(columns={'CASE','NAME','DATE_full'})
    case_count = 0
    with open(fname) as infile:
        #### 
        for line in infile:
            line_s = line.split()
            # If HEAD, keep the storm NAME and DATE/TIME
            if line_s[-1] == 'HEAD':
                NAME = line_s[0]
                DATE = line_s[1]
                CASE_TIME = line_s[2]
                # 
                DATE_fm = datetime.datetime.strptime(DATE,'%y%m%d')
                DATE_full = DATE_fm + pd.DateOffset(hours=int(CASE_TIME))
            # If LAST, add NAME and DATE, and add to CASE count
            elif (line_s[-1] == 'LAST'):
                df_new['NAME'] = NAME
                df_new['DATE_full'] = DATE_full
                if case_count == 0:
                    df_new['CASE'] = 0
                else:
                    df_new['CASE'] = case_count
                # SHIPS_all = SHIPS_all.append(df_new)
                case_count = case_count + 1
     #
            elif (line_s[-1] != 'HEAD') | (line_s[-1] != 'LAST'):
                line_len = len(line_s)
                # Predictors that start at time = -12 h
                if (line_len == 24):
                #
                    df_new[line_s[-1]] = line_s[0:-1]
    # For predictors with a number to the right of them (RSST, etc), add a variable that is RSST age (or whatever)
                elif line_len == 25:
                    df_new[line_s[-2]] = line_s[0:-2]
                    df_new[[line_s[-2]+'_AGE']] = line_s[-1]
              # Predictors that start at time = 0 h (need to pad first two rows)
                elif line_len == 22:
                    new_col = pd.Series(line_s[0:-1],name=line_s[-1])
                    new_col.loc[-1] = ""
                    new_col.loc[-2] = ""
                    new_col = new_col.sort_index().reset_index().drop(columns='index')
                    df_new[line_s[-1]] = new_col
              # Predictors that start at time = 0 h but also have a number to the right (age term)
                elif line_len == 23:
                 # For last, append everything to big file and move on
                    new_col = pd.Series(line_s[0:-2],name=line_s[-2])
                    new_col.loc[-1] = ""
                    new_col.loc[-2] = ""
                    new_col = new_col.sort_index().reset_index().drop(columns='index')
                    df_new[line_s[-2]] = new_col
                    df_new[[line_s[-2]+'_AGE']] = line_s[-1]
    return df_new


# ### `apply_scaling(df)` <a name="apply_scaling"></a>
# This function applies the scaling factors to various predictors in the `lsdiag` files. The scaling factors are contained in `SHIPS_factors.txt` (which should be located in the same directory as the `util` scripts), and are loaded using the `json` loader. Since initially nothing in the SHIPS files could be saved as a float (had to be saved as an int), the scaling factors are applied after the fact to get predictors in the right units. 
# 
# <b>Inputs:</b>
# * `df`: Dataframe containing contents of `lsdiag` file [Pandas dataframe]
# 
# <b>Outputs:</b>
# * `df`: Dataframe containing contents of `lsdiag` file after scaling is applied [Pandas dataframe]

# In[4]:


def apply_scaling_lsdiag(df):
    if 'LON_AGE' in df.columns:
        df = df.drop(columns={'LON_AGE'})
    # df.iloc[:,3:] = df.iloc[:,3:].apply(pd.to_numeric)
    df = df.apply(pd.to_numeric)
    with open(os.path.join(sys.path[0], "SHIPS_factors.txt"),"r") as f:
        SHIPS_factors = f.read()
    SHIPS_js = json.loads(SHIPS_factors)
    #
    col_names = df.columns
    for i_col in col_names:
        # print('feature is ',i_col)
        if i_col in SHIPS_js.keys():
            # print("yay")
            factor = SHIPS_js[i_col][0]
            # print('divide by ',factor)
            df[i_col] = df[i_col]/factor
        #else:
           # print("nay")
    return df


# ### `apply_landmask(df,mask_TYPE='SIMPLE_MASK')` <a name="apply_landmask"></a>
# This function calls `apply_land_mask` from `RT_SHIPS_preprocess.py`. `apply_land_mask` masks out forecasts that are too close to land. The default land mask is `SIMPLE_MASK`, which simply masks out any forecasts within a certain distance from land (generally, 100 km or less). 
# 
# <b>Inputs:</b>
# * `df`: Dataframe containing contents of `lsdiag` file (after scaling) [Pandas dataframe]
# * `mask_TYPE`: type of land masking to perform. Default is `SIMPLE_MASK` [str]
# 
# <b>Outputs:</b>
# * `SHIPS_mask`: land-masked Dataframe [Pandas dataframe]

# In[5]:


def apply_landmask_lsdiag(df,mask_TYPE='SIMPLE_MASK'):
    SHIPS_mask = apply_land_mask(df,mask_TYPE)
    return SHIPS_mask


# ### `get_dyn_predictors(df,BASIN,HR_first,HR_last,is_INTERP=True,calc_POT=False)`<a name="get_dyn_predictors"></a>
# This function calls `create_SHIPS_predictors_dyn` from `RT_SHIPS_preprocess.py`. `create_SHIPS_predictors_dyn` selects the specified dynamical predictors (here, we use generalized shear (`SHRG`), upper-level divergence (`D200`), lower-level vorticity (`Z850`), maximum potential intensity (`VMPI`), change in wind speed (`DELV`), mid-level relative humidity (`RHMD`), sea surface temperature (`RSST` or `NSST`, depending on basin), and ocean heat content (`RHCN` or `NOHC`, depending on basin). We also retain identifying information (`CASE`, `NAME`, `DATE_full`, `DTL`, `TIME`); and we muptiply `Z850` by -1 if we're making predictions in the Southern Hemisphere.  
# 
# <b>Inputs:</b>
# * `df`: Dataframe containing contents of `lsdiag` file after scaling/land masking [Pandas dataframe]
# * `BASIN`: Basin in which we are making our predictions (relevant for selection of `SST` and `OHC` variables, as West Pacific and S. Hem use different ones) [str]
# * `HR_first`: first forecast time (should almost always be -12, needed for persistence [int]
# * `HR_last`: last forecast time (24, 48, 72, etc depending on forecast) [int]
# * `is_INTERP`: will interpolation between missing values be performed? default is True [boolean]
# * `calc_POT`: will we calculate `POT` ($VMPI - I_0$) or use `MPI`? default is False [boolean]
# 
# <b>Outputs:</b>
# * `SHIPS_dyn_out`: Dataframe containing only dynamical predictors [Pandas dataframe]

# In[6]:


def get_dyn_predictors_lsdiag(df,BASIN,HR_first,HR_last,is_INTERP=True,calc_POT=False):
    #
    FORE_use = np.arange(HR_first,HR_last+1,6)
    #
    if ('NSST' in df.columns):# & ('NOHC' in df.columns):
        SST_sel = 'NSST'
        # OHC_sel = 'NOHC'
    else:
        SST_sel = 'RSST'
        #OHC_sel = 'RHCN'
    if 'NOHC' in df.columns:
        OHC_sel = 'NOHC'
    else:
        OHC_sel = 'RHCN'
    PREDICTORS_sel = ['CASE','NAME','DATE_full','DTL','TIME','SHRG','D200','Z850','VMPI','DELV','RHMD',SST_sel,OHC_sel]
    predictand_name = 'VMAX'
    #
    SHIPS_dyn_out = create_SHIPS_predictors_dyn(df,
                            PREDICTORS_sel,predictand_name,is_INTERP,FORE_use,calc_POT)
    # Multiply Z850 by =1 if in SH
    if BASIN == 'SOUTHERN_HEM':
        # print('multiply by -1 for SH')
        # print('before multiplying, Z850 mean is ',SHIPS_dyn_out['Z850'].mean())
        SHIPS_dyn_out['Z850'] = -1*SHIPS_dyn_out['Z850']
        # print('after multiplying, Z850 mean is ',SHIPS_dyn_out['Z850'].mean())
    # If we are in SH or WPac, rename RSST/RCHN to NSST/NOHC to match
    if ((BASIN == 'SOUTHERN_HEM') | (BASIN == 'WEST_PACIFIC')):
        SHIPS_dyn_out = SHIPS_dyn_out.rename(columns={'RSST':'NSST','RHCN':'NOHC'})
    # Rename DELV to DELV =12
    SHIPS_dyn_out = SHIPS_dyn_out.rename(columns={'DELV':'DELV -12'})
    return SHIPS_dyn_out


# ### `get_IR_predictors(df,HR_first,HR_last)`<a name="get_IR_predictors"></a>
# This function calls `create_SHIPS_predictors_IR` from `RT_SHIPS_preprocess.py`. `create_SHIPS_predictors_IR` selects the specified infrared (IR) predictors (here, we use GOES brightness temp (`GOES Tb`), the standard deviation of GOES brightness temp (`s(GOES Tb)`), the cold pixel percentage below -50C (`pct < -50 C`), and the storm size estimator (`storm size`), as well as the first for principal components of the IR imageray (`PC1`, `PC2`, `PC3`, and `PC4`). We also retain identifying information (`CASE`, `NAME`, `DATE_full`, `DTL`, `TIME`). Note that our first four quantities come from `IR00`, while the last four come from `PC00`. We use `IRM1` and `IRM3` (and `PCM1`/`PCM3` for the PC analysis), respectively, if an IR image is not available close to the forecast time. 
# 
# <b>Inputs:</b>
# * `df`: Dataframe containing contents of `lsdiag` file after scaling/land masking [Pandas dataframe]
# * `HR_first`: first forecast time (should almost always be -12, needed for persistence [int]
# * `HR_last`: last forecast time (24, 48, 72, etc depending on forecast) [int]
# 
# <b>Outputs:</b>
# * `SHIPS_IR_out`: Dataframe containing only IR predictors [Pandas dataframe]
# * `IR00_var_names`: list of strings containing IR variable names (needed to put dataframes back together) [list of strings]
# * `PC00_var_names`: list of strings containing PC variable names (needed to put dataframes back together) [list of strings]

# In[7]:


def get_IR_predictors_lsdiag(df,HR_first,HR_last):
    #
    FORE_use = np.arange(HR_first,HR_last+1,6)
    #
    predictors_sel_IR = ['CASE','NAME','DATE_full','TIME','IR00','IRM1','IRM3','PC00','PCM1','PCM3']
    # Identify time indices for desired IR variables (recall they are NOT time series)
    IR00_time_ind = [6,12,54,108]
    IR00_var_names = ['GOES Tb','s(GOES Tb)','pct < -50C','storm size']
    PC00_time_ind = [0,6,12,18]
    PC00_var_names = ['PC1','PC2','PC3','PC4']
    #
    warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)
    SHIPS_IR_out = create_SHIPS_predictors_IR(df,
                        predictors_sel_IR,FORE_use,IR00_time_ind,IR00_var_names,PC00_time_ind,PC00_var_names)
    return SHIPS_IR_out,IR00_var_names,PC00_var_names


# Example file directory and filename

# In[8]:


#ex_dir = '/home/mmcgraw/ML_for_TC_RI/VALIDATION_data/realtime/2021/'
# ex_dir ='/home/mmcgraw/ML_for_TC_RI/REALTIME/utils/test_files_realtime/'
# basin = 'AL'
# ex_file = '21091812AL'#_lsdiag.dat'


# In[9]:


# files_all = get_all_lsdiag(ex_dir,ex_file)


# ### `read_lsdiag_single(fname)`<a name="read_lsdiag_single"></a>
# This function reads in and does basic preprocessing on a single lsdiag file. It calls `read_in_lsdiag` to read in the file from a `.dat` format and save it as a Dataframe. Next, scaling (`apply_scaling`) and land masking (`apply_land_mask`) are applied. Note that if the storm has made landfall, we end our analysis here and simply return a NaN dataframe. Assuming we aren't over land, we extract dynamical and IR predictors (`get_dyn_predictors`) and (`get_IR_predictors`, respectively), and then combine both sets of predictors into a single dataframe. Finally, we get the ATCF ID (`BBNNYYYY`) for simplicity in future analysis.  
# 
# <b>Inputs:</b>
# * `fname`: full name and path corresponding to desired `lsdiag` file [str]
# 
# <b>Outputs:</b>
# * `df_ALL`: Dataframe containing preprocessed predictors from single `lsdiag` file [Pandas dataframe]

# In[10]:


def read_lsdiag_single(fname,basin,hr_last):
    df = read_in_lsdiag(fname)
    if df.empty:
        df_ALL = None
    else:
        df2 = df.set_index(['DATE_full','CASE','NAME'])
        # Apply SHIPS scaling factors (SHIPS predictors are not saved as floats)
        df_scaled = apply_scaling_lsdiag(df2)
        # Check to see if storm has made landfall. If yes, do not continue analysis. If no, apply the land mask (DTL>=100km)
        if df_scaled.reset_index().set_index(['TIME']).xs(0)['DTL'] <= 100:
            print('this storm has already made landfall')
            df_ALL = pd.DataFrame([np.nan]*3)
        else:
            print('still over the ocean')
            df_mask = apply_landmask_lsdiag(df_scaled.reset_index())
            # Get dynamical and IR predictors
            df_dyn_predictors = get_dyn_predictors_lsdiag(df_mask.reset_index(),basin,-12,hr_last,is_INTERP=True,calc_POT=False)
            df_IR_predictors,IR_var_names,PC_var_names = get_IR_predictors_lsdiag(df_mask.reset_index(),-12,hr_last)
            df_ALL = df_dyn_predictors
            df_ALL[IR_var_names] = df_IR_predictors[IR_var_names]
            df_ALL[PC_var_names] = df_IR_predictors[PC_var_names]
            df_ALL = df_ALL.drop(columns={'level_0','index'}).reset_index()
            df_ALL['ATCF ID'] = df_ALL['NAME'] + pd.to_datetime(df['DATE_full']).dt.year.astype(str)
    return df_ALL
    







