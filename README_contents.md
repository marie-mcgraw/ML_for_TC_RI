 #### Overview
`ML_for_TC_RI` trains one (or more) random forest models (as well as a simple logistic regression model) to predict whether or not tropical cyclone cases undergo rapid intensification (RI) using the SHIPS developmental data.  The SHIPS developmental dataset can be downloaded from the [RAMMB website] (https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/).  We currently analyze the East Pacific, West Pacific, Atlantic, and Southern Hemisphere. The code in this notebook reads in the SHIPS developmental dataset, performs some basic preprocessing steps (such as averaging predictors over the time period of interest, removing flags and missing data, scaling predictors, filtering cases based on distance from land, and so on), and trains the machine learning models.  We note that we are currently using the 30 kt/24 hour definition of rapid intensification for all ocean basins; however, the threshold and time period can be adjusted as desired. We train the machine learning models on SHIPS cases from 2005-2018. We combine a bootstrapping approach to training with a modified leave-one-year-out cross-validation scheme.  For each instance of model training, we complete the following steps: 
    1. Of the years 2005-2018, we randomly select 3 years to hold out for validation and train on the remaining years (including a hyperparameter sweep). 
    2. We validate the trained model using the 3-year validation sample.  We save the output of the validation, including the confusion matrix, the feature importances, the precision-recall curves, the classification report, the ROC curve, and the predicted classes for the validation sample. 
    3. We note that we train once, on samples from all ocean basins; during validation, we validate on a combined all-basin sample that includes cases from all basins, as well as on each basin individually.  

We repeat the process a total of 25 times, fully re-training the models every time.  We train each machine learning model on the same training instance simultaneously so that all models are trained and validated on the same years.  We take this approach because:
    1. When dealing with tropical cyclones, year to year variability is quite high, especially for rapid intensification which is by definition a rare event.  Thus, we don't want to overfit to any extreme years.
    2. Since rapid intensification can occur for only a single storm (or not at all) in a given year, we validate on 3 years instead of just 1 (and again, this helps us avoid overfitting)
    3. Our bootstrapped training approach gives us a built-in estimate of uncertainty, since we have 25 instances of training. 

After the bootstrapped model training process, we identify the best-guess model and train it once on the full training period (2005-2018). We then test this best-guess model with SHIPS cases from 2019-2021*. 

*As of 8/24/2022, the best-tracks for the JTWC basins (W. Pacific and S. Hemisphere) have not been released yet. Thus, the SHIPS developmental data contains the 2021 cases for the Atlantic and E. Pacific (NHC basins) but only goes through 2020 for the W. Pacific and S. Hemisphere.  We will update everything when the 2021 best-tracks are released. 

#### Scripts and Notebooks

<b>BASIC PREPROCESSING FOR SHIPS PREDICTORS</b>
1. <b>Read in SHIPS predictors from text files</b>
    1. <b>Input:</b> SHIPS developmental dataset [data] (located at https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/)
    2. <b>Code:</b> Scripts/SHIPS_reader_V2.py [Python script that reads in SHIPS files line by line and saves as CSV]
    3. <b>Output:</b> raw SHIPS data in csv [separate file per basin] (located in `DATA/processed/SHIPS_predictors_full_<basin>.csv`)
2. <b>Preprocess SHIPS predictors (scale, trim years/hours, remove missing data, select desired predictors)</b>
    1. <b>Input:</b> 
        1. raw SHIPS data in csv (from 1) (located in `DATA/processed/SHIPS_predictors_full_<BASIN>.csv`)
        2. SHIPS scaling factors (`SHIPS_factors.txt` home directory) 
    3. <b>Code:</b> 
        1. `preprocess_SHIPS_predictors.ipynb` [Jupyter notebook/Python script]
        2. `SHIPS_preprocess` utils [Python script in `util`]
    4. <b>Output:</b> preprocessed SHIPS predictors [separate file per basin] (located in `DATA/processed/SHIPS_processed_<basin>*.csv`)

<b>EXPLORATORY</b>
1.  Basic background data (# of samples per class and basin, etc): `SHIPS_dev_data_explore.ipynb` [Jupyter notebook]

<b>MACHINE LEARNING MODEL TRAINING</b>
1.  Train all machine learning models together.  Since we are doing a bootstrapped leave-n-years-out approach to training, we want to train all of our ML models simultaneously to ensure that each model is training on the same instance each time. 
    1. <b>Input:</b> preprocessed SHIPS predictors (`DATA/processed/SHIPS_processed_<basin>*.csv`; load each basin separately and combine all together) 
    2. <b>Code:</b>
        1. TRAIN_all_models_together [Jupyter notebook/Python script]
        2. SHIPS_ML_model_funcs utils [Python script in `util`] 
        3. SHIPS_preprocess utils [Python script in `util`]
        4. SHIPS_ML_model_funcs_imblearn utils [Python script in `util`] (ML functions for the `imblearn` version of `sklearn`)
        5. SHIPS_plotting utils [Python script in `util`]
    3. <b>Outputs:</b> 
        1. Save statistics/data from all bootstrapped training runs (logistic regression + random forest) (located in `DATA/ML_model_results/TRAINING/all_models_ROS_and_SMOTE`)
            1. Predicted Y values (contains our predictions of our target variables) [`predicted_Y_vals`]
            2. ROC / AUC values (ROC curves / AUC scores) [`ROC_AUC_vals`]
            3. Precision vs recall [`prec_vs_recall`]
            4. Feature importances (training) [`Feat_Imp_TRAIN`]
            5. Feature importances (validation) [`Feat_Imp_validation`]
            6. Confusion matrix [`Conf_Matrix`]
            7. Classification report [`Class_Report`]
        2. Basic model performance figures (all bootstrapped experiments) (located in `DATA/ML_model_results/TRAINING/all_models_ROS_and_SMOTE/figs/`).  Plots include:
            1.  Performance diagrams (curves and dots) and AUPD (area under PD curve) calculations
            2.  Precision vs recall curves
            3.  ROC plots
            4.  Box plots of AUC scores and contingency table quantities
            5.  Feature importances
            6.  Maximum CSI and frequency bias at max CSI
2.  Compare all ML models during training (plots)
    1. <b>Input:</b> ML model results (located in `DATA/ML_model_results/TRAINING/all_models_ROS_and_SMOTE`)
    2. <b>Code:</b>
        1. TRAIN_compare_RF_LR.ipynb [Jupyter notebook]
        2. SHIPS_plotting utils [Python script in `util`]
    3. <b>Outputs:</b> many plots (located in `Figures/TRAINING`)

<b>MACHINE LEARNING MODEL TESTING</b>
1. Test best-guess machine learning models on testing period
    1. <b>Input:</b> 
        1. ML model results (located in `DATA/ML_model_results/TRAINING/all_models_ROS_and_SMOTE`)
        2. preprocessed SHIPS predictors (load each basin separately and combine all together) (`DATA/processed/SHIPS_processed_<basin>*.csv`; load each basin separately and combine all together) 
    2. <b>Code:</b>
        1. `Test_SHIPS_2019-2021.ipynb` [Jupyter notebook/Python script]
        2. `SHIPS_ML_model_funcs` utils [Python script in `util`] 
        3. `SHIPS_preprocess` utils [Python script in `util`]
    3. <b>Outputs:</b> 
        1. Save best-guess ML model as a pickle (located in `DATA/ML_model_results/TESTING/all_models_ROS_and_SMOTE/`)
        2. Save statistics/data from model performance for all models on testing data (located in `DATA/ML_model_results/TESTING/all_models_ROS_and_SMOTE/`)
2. Compare LR and RF models during testing (plots)
    1. <b>Input:</b> ML model results (located in `DATA/ML_model_results/TESTING/all_models_ROS_and_SMOTE/`)
    2. <b>Code:</b>
        1. `TEST_compare_LR_RF.ipynb` [Jupyter notebook]
        2. `SHIPS_plotting` utils [Python script in `util`]
    3. <b>Outputs:</b> many plots (located in Figures/TESTING)
<b>VALIDATION WITH EXISTING PRODUCTS</b>
1. Read in best-track data for testing period 
    1. Input: ATCF best tracks for 2019-2021 (located at https://ftp.nhc.noaa.gov/atcf/archive/.  Select the year(s) of interest (2019, 2020, 2021 in this case) and acquire the b-decks (files that begin with `b`)). 
    2. Code: `read_in_best_track.ipynb` [Jupyer notebook/Python script] 
    3. Output: `best_tracks_{yr}.csv` in .csv format (located in `VALIDATION_data/processed/`)
2. Read in e-decks (probabilistic forecasts) for testing period 
    1. Input: Probabilistic intensity forecasts (e-decks) for testing period (located at https://ftp.nhc.noaa.gov/atcf/archive/)
        a. Note that prior to 2020, the probabilistic forecasts were not publicly available in ATCF format. Thus, reading in the e-deck data from 2019 uses a different script since the file is formatted differently. 
    2. Code: `get_edecks_2019.ipynb` [the 2019 e-decks] / `get_edecks_2020.ipynb` [2020/2021 edecks]
    3. Output: `etracks_RI_{year}.csv` in .csv format [probabilistic forecasts for RI cases] / `etracks_IN_{year}.csv` in .csv format [probabilistic forecasts for intensifying cases] (located in `VALIDATION_data/edecks/`)
3. Make reliability diagrams to compare our model with consensus RI forecast and SHIPS-RII.  Currently we only make this comparison for East Pacific and Atlantic (basins with SHIPS-RII). 
    1. <b>Inputs:</b> 
        1. Processed best-track data (`best_tracks_{year}.csv` located in `VALIDATION_data/processed/`)
        2. Processed e-deck data (`etracks_RI_{year}.csv` located in `VALIDATION_data/processed/edecks/`)
        3. Predicted storm intensity classes for 2019-2021 from ML model testing (`DATA/ML_model_results/TESTING/all_models_ROS_and_SMOTE/PREDICTED_Y_vals*.csv`)
    2. <b>Code:</b> 
        1. `SHIPS_reliability_diagrams_2019-2021.ipynb` [Jupyter notebook]
        2. `SHIPS_ML_model_funcs` utils [Python script in `util`] 
        3. `SHIPS_preprocess` utils [Python script in `util`]
        4. `SHIPS_plotting` utils [Python scrxipt in `util`]
    3. <b>Output:</b> Reliability diagrams for Atlantic and East Pacific located in `Figures/TESTING/`