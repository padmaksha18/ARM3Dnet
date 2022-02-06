
This repository contains the code for the following paper:
Deep diffusion-based forecasting of COVID-19 by incorporating network-level mobility information
P Roy, S Sarkar, S Biswas, F Chen, Z Chen, N Ramakrishnan, CT Lu

Modeling the spatiotemporal nature of the spread of infectious diseases can provide useful intuition in understanding the time-varying aspect of the disease spread and the underlying complex spatial dependency observed in people’s mobility patterns. Besides, the county level multiple related time series information can be leveraged to make a forecast on an individual time series. Adding to this challenge is the fact that real-time data often deviates from the unimodal Gaussian distribution assumption and may show some complex mixed patterns. Motivated by this, we develop a deep learning-based time-series model for probabilistic forecasting called Auto-regressive Mixed Density
Dynamic Diffusion Network (ARM3Dnet), which considers both people’s mobility and disease spread as a diffusion process on a dynamic directed graph. The Gaussian Mixture Model layer is implemented to consider the multimodal nature of the realtime data while learning from multiple related time series. We show that our model, when trained with the best combination of dynamic covariate features and mixture components, can outperform both traditional statistical and deep learning models in forecasting the number of Covid-19 deaths and cases at the county level in the United States.

# Steps to run the repo:

1. Run datapreprocess_time_series.py to preprocess the NYT covid data.

2. Run train_d3_armnet.ipynb to generate the dynamic covariates and also to train the model on the data.

3. Run evaluate.py to get the test results in the forecasting range.

Dynamic Covariates created for 6 months and stored in data folder. 
