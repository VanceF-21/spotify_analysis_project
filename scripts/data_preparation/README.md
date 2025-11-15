# data preparation

## Overview

This analysis utilizes the Spotify "Top 200" playlists dataset (2017-2023). The dataset comprises three main categories of variables: Audio attributes (e.g., danceability, energy, loudness), Artist attributes (e.g., nationality, continent, track ID), and Popularity metrics (e.g., pop_points_total). A robust preprocessing and feature engineering pipeline was executed to clean and prepare the data for analysis. Following this, Exploratory Data Analysis (EDA) was conducted to understand intrinsic data relationships, such as the strong correlation between Energy and Loudness ($\rho > 0.7$) and the weak linear relationship between Rank and audio features. The final preprocessed dataset contains 651,936 rows and 21 columns.

## Method

- **Data Preprocessing:**  The pipeline involved parsing CSVs, correcting typos, and ensuring data integrity by removing duplicates and rows with missing keys. We cleaned numerical data, corrected a Loudness anomaly, and clipped audio features to their theoretical [0,1] domain.
- **Feature Engineering:** We engineered new features: a normalized artist key, calendar fields (year, month, yyyymm), a clean rank column, and two harmonized popularity metrics (pop_points_total, pop_points_artist).
- **Exploratory Data Analysis:** : Correlation analysis (Fig. 3 in the paper) showed a strong positive correlation between Energy and Loudness ($\rho > 0.7$), and weak linear relationships between Rank and most audio features.
