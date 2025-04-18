#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import pandas as pd
import os, glob
# from google.colab import drive
import datetime


# In[2]:


# drive.mount('/content/drive')
ext = '.dat'
fdir_load = '/mnt/tc-ingest1/data/psdi_new/SHIPS/lsdiag/'
fdir_save = 'DATA/processed/realtime/'
# Make data directory if it doesn't exist yet
if not os.path.exists(fdir_save):
    os.makedirs(fdir_save)
yr_start = 2019
yr_end = 2021
yrs_all = np.arange(yr_start,yr_end+1)
fname_save = fdir_save+'SHIPS_realtime_predictors_{yr_start}-{yr_end}_ALL_basins.csv'.format(yr_start=yr_start,
                                                                    yr_end=yr_end)


# In[ ]:





# In[ ]:


"""Load file

If line is HEAD, set NAME to HEAD[0]; set DATE to HEAD[1], set CASE_TIME to HEAD[2]

Combine DATE and CASE_TIME

If line is LAST, increase case_count by 1

Else, read in case contents
"""

SHIPS_all = pd.DataFrame()
chunk_count = 0
case_count = 0
for iyr in yrs_all:
    fdir_yr = fdir_load+'{yr2}'.format(yr2=str(iyr)[-2:])
    fnames_all = glob.glob(fdir_yr+'*_lsdiag.dat')
    print('year: ',iyr)
    for fname in fnames_all:
        with open(fname, mode='r', encoding='utf-8-sig') as infile:
              for line in infile:
                line_s = line.split()
                # If HEAD, keep the storm NAME and DATE/TIME
                if line_s[-1] == 'HEAD':
                    NAME = line_s[0]
                    DATE = line_s[1]
                    CASE_TIME = line_s[2]
                    ATCFID = line_s[7]
                    print(ATCFID)
                    DATE_fm = datetime.datetime.strptime(DATE,'%y%m%d')
                    DATE_full = DATE_fm + pd.DateOffset(hours=int(CASE_TIME))
                    df_new = pd.DataFrame(columns={'CASE','NAME','DATE_full','ATCFID'})
                #
                elif (line_s[-1] == 'LAST'):
                    df_new['NAME'] = NAME
                    df_new['DATE_full'] = DATE_full
                    df_new['ATCF Basin'] = ATCFID[0:2]
                    df_new['ATCFID'] = ATCFID
                    if case_count == 0:
                        df_new['CASE'] = 0
                    else:
                        df_new['CASE'] = case_count
                    SHIPS_all = SHIPS_all.append(df_new)
                    case_count = case_count + 1
                    if np.mod(case_count,200) == 0:
                        SHIPS_all.to_csv(fname_save)
                        print('saved at case ',case_count)
                    del df_new
                    print('on to case ',case_count)
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
    # 
SHIPS_all.to_csv(fname_save)


# In[31]:


SHIPS_all.groupby(['ATCF Basin','ATCFID']).count()


# In[ ]:




