#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import os, glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Get names for best track (b-deck) files, organized by year

# In[10]:


year_sel = 2019
bdeck_dir = 'VALIDATION_data/bdecks/{year_sel}/'.format(year_sel=year_sel)
fnames_all = [os.path.join(bdeck_dir, f) for f in os.listdir(bdeck_dir) if f.endswith('.dat')]


# #### Format for b-deck files
# B-deck is the best track information for tropical cyclones.  File names are bBBCCYYYY.dat.  
# * b: b-deck
# * BB: basin (al is Atlantic, ep is East Pacific, cp is Central Pacific, sl is South Atlantic)
# * CC: storm number. 01-30, numbered storms, not recycled.  80-89: internal training, IGNORE. 90-99: invest, areas of interest, redeployed as needed. A0-Z9: recycled invest series (?). Come on NOAA god
# * YYYY: Year. 
# 
# 
# Full README is at: https://www.nrlmry.navy.mil/atcf_web/docs/database/new/database.html
# 

# ##### What's in the file?
# BASIN, CY, DATE [YYYYMMDDHH], TECNHUM/MIN, TECH, TAU, LAT N/S, LON E/W, VMAX, MSLP, TY, RAD, WINDCODE, RAD1, RAD2, RAD3, RAD4, POUTER, ROUTER, RMW, GUSTS, EYE, SUBREGION, MAXSEAS, INITIALS, DIR, SPEED, STORMNAME, DEPTH, SEAS, SEASCODE, SEAS1, SEAS2, SEAS3, SEAS4, USERDEFINED, userdata
# 
# <b>Parameters we care about:</b>
# * BASIN: Basin (EP, AL, CP)
# * CY: cyclone number, restarts every year
# * DATE: date, in YYYYMMDDHH
# * TECH: acronym for objective technique
# * TAU: forecast period -24-240 hours
# * LAT N/S: Latitude, 0-900 (0.1 degrees); N/S for hemisphere
# * LON E/W: Longitude, 0-1800 (0.1 degrees); E/W for hemisphere
# * VMAX: maximum sustained wind speed, 0-300 kts
# * MSLP: minimum sea level pressure (850-1050 mb)
# * TY: highest level of TC disturbance
# * STORMNAME: literal storm name, number, NONAME or INVEST, or TCcyx where: cy = Annual cyclone number 01 - 99; x  = Subregion code: W,A,B,S,P,C,E,L,Q.

# <b>Parameters we (probably) don't care about as much: </b>
# * TECHNUM/MIN: objective technique sorting number, minutes for best track: 00 - 9
# * RAD: wind intensity for radii defined in this record (34, 50, 64 kt)
# * WINDCODE: Radius code, AA - full circle; NEQ, SEQ, SWQ, NWQ - quadrant 
# * RAD1: If full circle, radius of specified wind intensity, or radius of first quadrant wind intensity as specified by WINDCODE.  0 - 999 n mi
# * RAD2: If full circle this field not used, or radius of 2nd quadrant wind intensity as specified by WINDCODE.  0 - 999 n mi.
# * RAD3: If full circle this field not used, or radius of 3rd quadrant wind intensity as specified by WINDCODE.  0 - 999 n mi.
# * RAD4: If full circle this field not used, or radius of 4th quadrant wind intensity as specified by WINDCODE.  0 - 999 n mi.
# * POUTER: pressure in millibars of the last closed isobar, 900 - 1050 mb.
# * ROUTER: radius of the last closed isobar, 0 - 999 n mi.
# * RMW: radius of max winds, 0 - 999 n mi.
# * GUSTS: gusts, 0 - 999 kt.
# * EYE: eye diameter, 0 - 120 n mi.
# * SUBREGION: subregion code: W,A,B,S,P,C,E,L,Q. A: Arabian Sea, B: Bay of Bengal, C: Central Pacific, E: East Pacific, L: Atlantic, P: South Pacific, Q: South Atlantic, S: South Indian Ocean, W: West Pacific
# * MAXSEAS: max seas, 0 - 999 ft.
# * INITIALS: Forecaster's initials used for tau 0 WRNG or OFCL, up to 3 chars.
# * DIR: storm direction, 0 - 359 degrees.
# * SPEED: storm speed, 0 - 999 kts.
# * DEPTH: system depth, D - deep, M - medium, S - shallow, X - unknown
# * SEAS: Wave height for radii defined in SEAS1 - SEAS4, 0 - 99 ft.
# * SEASCODE: Radius code: AAA - full circle; NEQ, SEQ, SWQ, NWQ - quadrant 
# * SEAS1: first quadrant seas radius as defined by SEASCODE,  0 - 999 n mi.
# * SEAS2: second quadrant seas radius as defined by SEASCODE, 0 - 999 n mi.
# * SEAS3: third quadrant seas radius as defined by SEASCODE,  0 - 999 n mi.
# * SEAS4: fourth quadrant seas radius as defined by SEASCODE, 0 - 999 n mi.

# In[11]:


b_deck_ALL = pd.DataFrame()


# In[12]:


for i_line in np.arange(0,len(fnames_all)):
    print('reading ',fnames_all[i_line])
    lines = open(fnames_all[i_line]).readlines()
    b_deck = pd.DataFrame(columns=['BASIN','CYCLONE NO','DATE','TECHNUM','TECH','TAU','LAT','LON','VMAX','MSLP','TYPE',
                              'RAD','WINDCODE','RAD1','RAD2','RAD3','RAD4','P Outer','R Outer','RMW','GUSTS','EYE',
                              'SUBREGION','MAXSEAS','INITIALS','DIR','SPEED','NAME','DEPTH','SEAS','SEASCODE',
                              'SEAS1','SEAS2','SEAS3','SEAS4'],
                     index = np.arange(0,len(lines)))
    for i_sub in np.arange(0,len(lines)):
        #
        i_sel = lines[i_sub].split()
        max_len = min(len(i_sel),35)
        b_deck.iloc[i_sub,0:max_len] = i_sel[0:max_len]
    b_deck_ALL = b_deck_ALL.append(b_deck)


# Remove superfluous commas

# In[13]:


b_deck_ALL = b_deck_ALL.reset_index().replace(",","",regex=True)


# In[14]:


b_deck_ALL['DATE'] = pd.to_datetime(b_deck_ALL['DATE'].astype(str),format='%Y%m%d%H')
#
ATCFID = b_deck_ALL['BASIN']+b_deck_ALL['CYCLONE NO']+str(year_sel)
b_deck_ALL['ATCF ID'] = ATCFID


# In[15]:


b_deck_ALL['TIME'] = b_deck_ALL['DATE'].dt.hour
b_deck_ALL = b_deck_ALL.drop(columns='index')

bdeck_save = 'VALIDATION_data/processed/best_tracks_{year_sel}.csv'.format(year_sel=year_sel)
b_deck_ALL.to_csv(bdeck_save)

b_deck_ALL.iloc[50]


# In[16]:


b_deck_ALL['BASIN'].unique()


# In[ ]:




