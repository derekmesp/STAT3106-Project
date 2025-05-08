## Derek and Lloxci's STAT3106 Final Project

# SRC
Contains .py files:
- `ingest_data.py`: Removes duplicates and null values from eviction data. Takes in raw CSV file and outputs a cleaned CSV file for use in downstream analysis.
- `eda.py`: Exploratory data analysis containing EvictionVisualizer class. Capable of plotting static folium graphs, time series folium graphs, a graph of evictions per borough over time, and produces a dataframe containing eviction data grouped by binned latitude/longitude.
- `feat_gen.py`: Transform eviction data into a pivot table containing every unique binned lat_bin, lon_bin pair as a row and each unique date in the dataset as a column. The values of the dataframe are the number of evictions at that lat_bin lon_bin pair on that specific day.
- `model_creation.py`: Contains methods for creating Bayesian Gaussian Mixture, LogisticRegression, RandomForest models and outputs the trained model (along with classification summaries for LogisticRegression and RandomForest). Processes features such that the lat_bin, lon_bin pairs are weighted by the total number of evictions at that area.
- `model_eval.py`: Contains methods for plotting time series of specific clusters, demographics per eviction risk level, demographics for clusters, and contains method for doing a t-test with FDA adjusted p-values for the clusters with the highest and lowest number of evictions.
- `report.ipynb`: Notebook containing all downstream analysis.

# data
Contains raw eviction data collected from [NYC Open Data](https://data.cityofnewyork.us/City-Government/Evictions/6z8x-wfk4/about_data) and preprocessed CSV file. Also contains data for demographics for each NTA collected from [NYC Department of Aging](https://www.nyc.gov/assets/dfta/downloads/pdf/reports/Demographics_by_NTA.pdf).

# Visualizations
Contains interactive folium visualizations in html format. GitHub does not do well with displaying interactive plots so the files would have to opened in a new tab.
