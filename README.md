# anomaly_detection
Supplementary source code for the article with doi:10.20944/preprints202503.1358.v1
This is a collection of Python scripts performing various operations described in the above mentioned article. 
Everything starts with downloading the Aruba and HH120 datasets from CASAS (https://data.casas.wsu.edu/download/).
The preprocessing is executed with preproc_aruba.py and preproc_HH120.py.
Time series are created in two steps (create_vectors_aruba.py, create_vectors2_aruba.py. Same for HH120.)
split_v2_with_parameter.py splits the time series in train.csv and test.csv.
Several prediction models are implemented (RF.py - random forest, svr.py - support vector regression, sarima.py - Seasonal ARIMA, etc.)
The results described in the article are obtained with Prophet_fuzzy_E_tuned.py.
