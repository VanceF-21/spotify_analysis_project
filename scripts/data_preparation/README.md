# data preparation

## Overview

This analysis uses the Spotify "Top 200" dataset (2017-2023), which includes Audio attributes, Artist attributes, and Popularity metrics. Data was cleaned via preprocessing and feature engineering. Exploratory Data Analysis (EDA) revealed a strong correlation between Energy and Loudness ($\rho > 0.7$) but weak linear relationships between Rank and audio features. The final dataset contains 651,936 rows and 21 columns.
## Method

- **Data Preprocessing:**  The pipeline involved parsing CSVs, correcting typos, and ensuring data integrity by removing duplicates and rows with missing keys. We cleaned numerical data, corrected a Loudness anomaly, and clipped audio features to their theoretical [0,1] domain.
- **Feature Engineering:** We engineered new features: a normalized artist key, calendar fields (year, month, yyyymm), a clean rank column, and two harmonized popularity metrics (pop_points_total, pop_points_artist).
- **Exploratory Data Analysis:** : Correlation analysis (Fig. 3 in the paper) showed a strong positive correlation between Energy and Loudness ($\rho > 0.7$), and weak linear relationships between Rank and most audio features.
