<b>BASIC PREPROCESSING FOR SHIPS PREDICTORS</b>
1. <b>Read in SHIPS predictors from text files</b>
    1. <b>Input:</b> SHIPS developmental dataset [data] (located in DATA/SHIPS_data)
    2. <b>Code:</b> SHIPS reader [Python script]
    3. <b>Output:</b> raw SHIPS data in csv [separate file per basin] (located in `DATA/processed/SHIPS_predictors_full_<basin>.csv`)
2. <b>Preprocess SHIPS predictors (scale, trim years/hours, remove missing data, select desired predictors)</b>
    1. <b>Input:</b> 
        1. raw SHIPS data in csv (from 1) (located in `DATA/processed/SHIPS_predictors_full_<BASIN>.csv`)
        2. SHIPS scaling factors (`SHIPS_factors.txt` file) 
    3. <b>Code:</b> 
        1. `preprocess_SHIPS_predictors.ipynb` [Jupyter notebook/Python script]
        2. SHIPS_preprocess utils [Python script in `util`]
    4. <b>Output:</b> preprocessed SHIPS predictors [separate file per basin] (located in `DATA/processed/SHIPS_processed_<basin>*.csv`)

<b>EXPLORATORY</b>
1.  Sampling strategies
2.  Basic background data (# of samples per class and basin, etc)

<b>MACHINE LEARNING MODEL TRAINING</b>
1.  Train basic logistic regression model
    1. <b>Input:</b> preprocessed SHIPS predictors (`DATA/processed/SHIPS_processed_<basin>*.csv`; load each basin separately and combine all together) 
    2. <b>Code:</b>
        1. TRAIN_regression_model_RI_no_RI [Jupyter notebook/Python script]
        2. SHIPS_ML_model_funcs utils [Python script in `util`] 
        3. SHIPS_preprocess utils [Python script in `util`]
        4. SHIPS_plotting utils [Python script in `util`]
    3. <b>Outputs:</b> 
        1. Save statistics/data from all bootstrapped training runs (located in `DATA/ML_model_results/TRAINING/LOGISTIC/`)
            1. Predicted Y values (contains our predictions of our target variables) (`predicted_Y_vals`)
            2. ROC / AUC values (ROC curves / AUC scores) (`ROC_AUC_vals`)
            3. Precision vs recall [`prec_vs_recall`]
            4. Feature importances (training) [`Feat_Imp_TRAIN`]
            5. Feature importances (validation) [`Feat_Imp_validation`]
            6. Confusion matrix [`Conf_Matrix`]
            7. Classification report [`Class_Report`]
        2. Basic model performance figures (all bootstrapped experiments) (located in `DATA/ML_model_results/TRAINING/LOGISTIC/figs/`).  Plots include:
            1.  Performance diagrams (curves and dots) and AUPD (area under PD curve) calculations
            2.  Precision vs recall curves
            3.  ROC plots
            4.  Box plots of AUC scores and contingency table quantities
            5.  Feature importances
            6.  Maximum CSI and frequency bias at max CSI
2.  Train basic random forest model
    1. <b>Input:</b> preprocessed SHIPS predictors (`DATA/processed/SHIPS_processed_<basin>*.csv`; load each basin separately and combine all together) 
    2. <b>Code:</b>
        1. TRAIN_random_forest_model_RI_no_RI [Jupyter notebook/Python script]
        2. SHIPS_ML_model_funcs utils [Python script in `util`] 
        3. SHIPS_preprocess utils [Python script in `util`]
        4. SHIPS_plotting utils [Python script in `util`]
    3. <b>Outputs:</b> 
        1. Save statistics/data from each bootstrapped training run (located in `DATA/ML_model_results/TRAINING/RF/`)
            1. Predicted Y values (contains our predictions of our target variables) (`predicted_Y_vals`)
            2. ROC / AUC values (ROC curves / AUC scores) (`ROC_AUC_vals`)
            3. Precision vs recall [`prec_vs_recall`]
            4. Feature importances (training) [`Feat_Imp_TRAIN`]
            5. Feature importances (validation) [`Feat_Imp_validation`]
            6. Confusion matrix [`Conf_Matrix`]
            7. Classification report [`Class_Report`]
        2. Basic model performance figures (all bootstrapped experiments) (located in `DATA/ML_model_results/TRAINING/RF/figs/`).  Plots include:
            1.  Performance diagrams (curves and dots) and AUPD (area under PD curve) calculations
            2.  Precision vs recall curves
            3.  ROC plots
            4.  Box plots of AUC scores and contingency table quantities
            5.  Feature importances
            6.  Maximum CSI and frequency bias at max CSI
3.  Compare LR and RF models during training (plots)
    1. <b>Input:</b> ML model results (2 LR models + RF model) (located in `DATA/ML_model_results/TRAINING`)
    2. <b>Code:</b>
        1. Compare RF and LR training [Jupyter notebook/Python script]
        2. SHIPS_plotting utils [Python script in `util`]
    3. <b>Outputs:</b> many plots (located in FIGURES/TRAINING)

<b>MACHINE LEARNING MODEL TESTING</b>
1. Test best-guess logistic regression model on testing period
    1. <b>Input:</b> 
        1. ML model results (do each LR model separately) (located in `DATA/ML_model_results`)
        2. preprocessed SHIPS predictors (load each basin separately and combine all together) (located in `DATA/processed`)
    2. <b>Code:</b>
        1. Test simple model RI vs no RI [Jupyter notebook/Python script]
        2. SHIPS_ML_model_funcs utils [Python script in `util`] 
    3. <b>Outputs:</b> 
        1. Save best-guess ML model (located in `DATA/trained_models`)
        2. Save statistics/data from model performance on testing data (located in `DATA/ML_model_results/testing`)
2. Test best-guess random forest model on testing period
    1. <b>Input:</b> 
        1. ML model results (located in `DATA/ML_model_results`)
        2. preprocessed SHIPS predictors (load each basin separately and combine all together) (located in `DATA/processed`)
    2. <b>Code:</b>
        1. Test RF model RI vs no RI [Jupyter notebook/Python script]
        2. SHIPS_ML_model_funcs utils [Python script in `util`] 
    3. <b>Outputs:</b> 
        1. Save best-guess ML model (located in `DATA/trained_models`)
        2. Save statistics/data from model performance on testing data (located in `DATA/ML_model_results/testing`)
3. Compare LR and RF models during testing (plots)
    1. <b>Input:</b> ML model results (2 LR models + RF model) (located in `DATA/ML_model_results/testing`)
    2. <b>Code:</b>
        1. Compare RF and LR testing [Jupyter notebook/Python script]
        2. SHIPS_plotting utils [Python script in `util`]
    3. <b>Outputs:</b> many plots (located in FIGURES/TESTING)
<b>VALIDATION WITH EXISTING PRODUCTS</b>
1. Read in best-track data for testing period 
    1. Input: b-decks for testing period (located in `DATA/bdecks`)
    2. Code: read in b-decks
    3. Output: b-decks in .csv (located in `DATA/bdecks`)
2. Read in e-decks (probabilistic forecasts) for testing period 
    1. Input: e-decks for testing period (located in `DATA/edecks`)
    2. Code: read in e-decks
    3. Output: e-decks in .csv (`located in DATA/edecks`)
3. Make reliability diagrams to compare our model with consensus, SHIPS-RII
    1. Inputs: b-decks, e-decks, testing
    2. Code: reliability diagram
    3. Output: plots
