#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import pandas as pd
import datetime
import glob


# #### Get list of all files for desired years
# Note: `fdir` is pointing to `tcnas01`, where Kate mounted the RIPA edecks

# In[2]:


yr_sel = [2019,2020,2021]
fdir = '/mnt/tcnas01/musgrave/ryan_ridata/RIPA/RIPA/'
all_filesx = []
#fname_test = '20072600AL9220_ships.txt'
for iyr in yr_sel:
    ifiles = glob.glob(fdir+'e*{yr}.dat'.format(yr=iyr))
    all_filesx = all_filesx + ifiles
no_files = len(all_filesx)
fname_test = all_filesx[1]


# In[3]:


all_files=[x for x in all_filesx if len(x)<=57]


# #### Read in each file one at a time
# 
# #### General Format of Files
# BASIN  CYCLONE NUMBER  DATE (YYYYMMDDHH)   ProbFormat   Tech   TAU    LAT N/S    Lon E/W    PROB
# 
# ProbFormat: 
# * TR: track
# * IN: intensity
# * RI: rapid intensification
# * RW: rapid weakening
# * WR: wind radii
# * PR: pressure
# * GN: TC genesis probability
# * GS: TC genesis shape
# * ER: eyewall replacement
# 
# Tech: objective technique
# 
# TAU: forecast period (0-168 hrs)
# 
# Lat N/S: 0-900 [tenths of degrees]
# 
# Lon E/W: 0-1800 [tenths of degrees]
# 
# Prob:  probability of ProbItem, 0-100%
# 
# ##### Intensity Probability
# * ProbItem: Wind speed (bias adjusted), 0-300 kts
# * TY:  level of TC development; currently unused
# * Half_Range: Half the probability range (radius), 0-50 kts
# 
# ###### Rapid Intensification Probability / Rapid Weakening Probability
# * ProbItem: Intensity change, 0-300 kts
# * V: final intensity, 0-300 kts
# * Initials: forecaster initials
# * RIstartTAU: RI start time, 0-168 hours
# * RIstopTAU: RI stop time, 0-168 hrs

# In[4]:


with open(fname_test) as fn:
    for line in fn:
        #if line.startswith("TIME (HR)"):
        
        print(line)


# In[6]:


IN_valid_names = {'ATCF BASIN':[],'CYCLONE NO':[],'DATE':[],'ProbFormat':[],
                'Tech':[],'TAU':[],'LAT':[],'LON':[],'ProbItem':[],'TY':[],
                            'foo':[],'HalfRange':[]}
RI_valid_names = {'ATCF BASIN':[],'CYCLONE NO':[],'DATE':[],'ProbFormat':[],
                'Tech':[],'TAU':[],'LAT':[],'LON':[],'ProbItem':[],'Intensity Change':[],
                            'V':[],'Initials':[],'RIstartTAU':[],'RIstopTAU':[]}
etracks_valid_RI = pd.DataFrame.from_dict(RI_valid_names,orient='columns')
#etracks_valid_IN = pd.DataFrame.from_dict(IN_valid_names,orient='columns')
etracks_valid_RW = pd.DataFrame.from_dict(RI_valid_names,orient='columns')


# In[ ]:





# In[7]:


for i in np.arange(0,len(all_files)):
    fname = all_files[i]
    print('reading ',fname)
    #i_basin = BASIN[i]
    fname_full = open(fname)
    lines = fname_full.readlines()

    for iline in np.arange(0,len(lines)):
        #iline = 37
        line_sel = lines[iline].split()
        i_pf = line_sel[3]
        # print(i_pf)
        if i_pf == 'RI,':
            #print('RI!')
            etracks_valid_RI = etracks_valid_RI.append(pd.Series(line_sel, index = etracks_valid_RI.columns),ignore_index=True)
       # elif i_pf == 'IN,':
            #print('Intensity')
            #etracks_valid_IN = etracks_valid_IN.append(pd.Series(line_sel, index = etracks_valid_IN.columns),ignore_index=True)
        #
        elif i_pf == 'RW,':
            print('RW')
            etracks_valid_RW = etracks_valid_RW.append(pd.Series(line_sel, index = etracks_valid_RW.columns),ignore_index=True)


# ### Clean up data a bit.  
# * Remove extraneous commas
# * Replace basin names with full names
# * Convert dates into datetime format

# In[8]:


etracks_valid_RI = etracks_valid_RI.replace(",", "", regex=True)
etracks_valid_RW = etracks_valid_RW.replace(",", "", regex=True)
#etracks_valid_IN = etracks_valid_IN.replace(",","",regex = True)


# In[9]:


etracks_valid_RI['DATE'] = pd.to_datetime(etracks_valid_RI['DATE'].astype(str),format='%Y%m%d%H')
etracks_valid_RW['DATE'] = pd.to_datetime(etracks_valid_RW['DATE'].astype(str),format='%Y%m%d%H')
#etracks_valid_IN['DATE'] = pd.to_datetime(etracks_valid_IN['DATE'].astype(str),format='%Y%m%d%H')
etracks_valid_RI['YEAR'] = pd.to_datetime(etracks_valid_RI['DATE']).dt.year


# In[11]:


# ATCF ID
etracks_valid_RI['ATCFID'] = etracks_valid_RI['ATCF BASIN']+etracks_valid_RI['CYCLONE NO'] + etracks_valid_RI['YEAR'].astype(str)


# In[12]:


# Change basin names
etracks_valid_RI['BASIN'] = etracks_valid_RI['ATCF BASIN'].replace("AL","ATLANTIC",regex=True)
etracks_valid_RI['BASIN'] = etracks_valid_RI['ATCF BASIN'].replace("EP","EAST_PACIFIC",regex=True)
etracks_valid_RI['BASIN'] = etracks_valid_RI['ATCF BASIN'].replace("CP","CENTRAL_PACIFIC",regex=True)
etracks_valid_RI['BASIN'] = etracks_valid_RI['ATCF BASIN'].replace("WP","WEST_PACIFIC",regex=True)
etracks_valid_RI['BASIN'] = etracks_valid_RI['ATCF BASIN'].replace("SH","SOUTHERN_HEM",regex=True)
etracks_valid_RI['BASIN'] = etracks_valid_RI['ATCF BASIN'].replace("IO","INDIAN_OCEAN",regex=True)
#


# In[ ]:


etracks_valid_RI.to_csv('VALIDATION_data/edecks/RIPA_etracks_RI_{yr0}-{yr1}.csv'.format(yr0=yr_sel[0],
                                                                                       yr1=yr_sel[-1]))


# In[ ]:


import seaborn as sns
etracks_test = etracks_valid_RI.copy()
etracks_test['ProbItem'] = etracks_test['ProbItem'].astype(int)
sns.histplot(data=etracks_test,x='ProbItem',hue='BASIN')


# In[ ]:


foo = etracks_test.where((etracks_test['RIstopTAU'].astype(int) <= 24)
                        & (etracks_test['Intensity Change'].astype(int) > 0)).dropna(how='all')


# In[ ]:


sns.histplot(data=foo,x='ProbItem',binwidth=10,hue='BASIN')


# In[ ]:


etracks_test[(etracks_test['ProbItem'] > 70) & (etracks_test['Intensity Change'].astype(int)==40) & 
            (etracks_test['BASIN']=='EAST_PACIFIC')]


# In[ ]:




