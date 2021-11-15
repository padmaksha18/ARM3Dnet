# Steps to run spatio-temporal-covid-forecasting_2021 repo:

1. Run datapreprocess_time_series.py to preprocess the NYT covid data.

2. Run train_d3_armnet.ipynb to generate the dynamic covariates and also to train the model on the data.

3. Run evaluate.py to get the test results in the forecasting range.

Dynamic Covariates created for 6 months and stored in data folder. 
