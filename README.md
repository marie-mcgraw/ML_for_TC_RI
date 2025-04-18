# Machine Learning Model for Tropical Cyclone Rapid Intensification Forecasting
## Overview
![overview of SHIPS RI model](RI_exp_des.png)

This repository contains the machine learning model for tropical cyclone rapid intensification forecasting. This model was trained using the SHIPS Developmental Dataset (available at RAMMB: (https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/), and following conventions established by operational statistical hurricane intensity forecast models at the National Hurricane Center. A modified version of this model is currently in the research-to-operations pipeline at the National Hurricane Center. 

**Rapid intensification** refers to large changes in tropical cyclone intensity in short periods of time; in this model, we use the 30 kt/24 hour threshold established by Kaplan and DeMaria (1999). Rapid intensification, or RI, is by definition a rare event (occurring in approximately 5% of tropical cyclone forecasts), and is difficult to forecast. RI events are also frequently high-consequence, especially when rapid intensification occurs very closely to landfall. This model was developed with the goal of providing an additional data-driven intensity forecasting model that would improve the performance of the National Hurricane Center's **consensus** rapid intensification forecast. 

Since we are forecasting a rare event, and because tropical cyclone variability from year-to-year and across basins is quite large, we train our model using a bootstrapped model training process (see Figure 1, above). This process will be explained in more detail shortly. 

## Setup
After you have cloned this repo using `git clone`, navigate to the correct directory:
1. Set up the environment file using `conda`: `conda env create -f environment.yml` (note that creating the environment might take a few minutes; don't panic!).
2. Activate the environment using `conda activate SHIPS`. You can deactivate the environment using `conda deactivate`.
3. Run the setup script to create appropriate directories: `python setup.py`

## Data Ingest and Preprocessing
The model is trained on the SHIPS developmental dataset, which is publicly available at RAMMB (https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/). You can download the file yourself to get the most up to date version. Each basin has its own file. The SHIPS model was first created in the 1990s, so it's in a slightly old-fashioned format. We need to read it in line by line. To do that, you can run `SHIPS_reader_v2.py`. This script reads in the SHIPS data line by line, and saves it in a `.csv` format which will be much nicer to work with. Once we've done that, we can run `preprocess_SHIPS_predictors.ipynb` to create our training dataset. Note that we'll need `SHIPS_factors.txt` to scale the data appropriately. `preprocess_SHIPS_predictors.ipynb` performs some feature selection, land masking, and other quality control functions. 

`get_realtime_lsdiags.py` and `preprocess_SHIPS_predictors_REALTIME.ipynb` read in and preprocess the SHIPS realtime data files, respectively. 

*A note about realtime evaluation:* The SHIPS developmental model is a perfect prognostic model, and it's updated at the end of every season after the best-track tropical cyclone data has been released. Thus, SHIPS developmental data is not available in realtime. Since we ultimately want this model to run in realtime at the National Hurricane Center, we evaluate the model on both developmental data and on the realtime data that it would be ingesting in realtime. While the developmental data is publicly available, the realtime data isn't. Some of the realtime code is included in this repo, but you won't be able to run it unless you have your own access to the realtime data. 

Finally, `SHIPS_dev_data_explore.ipynb` is a notebook that explores the SHIPS developmental dataset for people who are not as familiar with statistical forecasting of tropical cyclone intensity. This notebook shows some visualizations such as number of rapid intensification events per basin per year, and intensity change forecast distributions. 
