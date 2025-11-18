# Music Style and Popularity Analysis on Spotify

This project investigates the interplay between musical style, contextual factors, and song popularity using the Spotify "Top 200" playlists dataset (2017–2023). The analysis employs a two-stage framework combining unsupervised clustering and supervised classification to translate low-level audio features into actionable insights for music industry stakeholders.

## Overview

The project implements a two-stage analytical pipeline:

1. **Task 1: Music Style Classification** - Identifies representative music styles through K-Means++ clustering on audio features
2. **Task 2: Popularity Prediction** - Uses cluster-based features to predict song popularity and identify key success factors

## Project Structure

```
spotify_analysis_project/
├── data/
│   ├── data_preparation/
│   │   ├── kmeans_clustered_data.csv
│   │   ├── spotify_cleaned_data.csv
│   │   ├── data_clean_new.csv
│   │   ├── spotify_dedup_by_day_title__with_combo_stats.csv
│   │   └── Spotify_Dataset_V3.csv
│   ├── task1/ 
│   │   │── spotify_database.db
│   │   └── spotify_data_V3.csv   
│   └── task2/
│       │── spotify_dataset.csv
│       │── spotify_dataset_sample.csv  
│       └── data_with_famous_artist.csv
├── scripts/
│   ├── data_preparation/
│   │   ├── correlation_heatmap.py          # Correlation heatmap
│   │   ├── spotify_data_cleaning.py        # Data cleaning
│   │   └── table1.py
│   ├── music_style_classification/
│   │   ├──Kmeans_stability_analysis.py
│   │   ├──MAIN_Elbow_Point_Optimal_Clustering.py
│   │   ├──SectionB Kmeans_feature_and_dimension.py
│   │   ├──Hierarchical clustering_result
│   │   ├──Kmeans++8_Feature_Mean_Std
│   │   └──visualisation
│   └── feature_popularity_analysis/
│       ├── classification_binary.py                 # Main classification script
│       ├── regression.py                            # Regression baseline (optional)
│       ├── figs.ipynb
│       ├── figs                                     # For figures
│       └── results                                  # For results
└── README.md
```

## Dataset

The dataset contains Spotify "Top 200" playlists data spanning 2017–2023 with the following key features:

**Audio Attributes:**
- Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Valence

**Contextual Variables:**
- Nationality, Continent, Season, Weekend indicator (is_weekend)

**Popularity Metrics:**
- Pop_points_total, Pop_points_artist, Rank, popularity_class


## Download the Datasets

You can download datasets used in this project from Hugging Face before running any code. All scripts in Task 1 and Task 2 depend on these files being placed in the correct directory structure.

Because GitHub cannot host large files, all datasets required for this project must be downloaded manually from Hugging Face.

👉 **Hugging Face Dataset Link:**  https://huggingface.co/datasets/Frankieeee21/spotify-analysis-dataset

The dataset includes raw, cleaned, and intermediate files used across both tasks:

- `Spotify_Dataset_V3.csv`
- `spotify_cleaned_data.csv`
- `kmeans_clustered_data.csv`
- `spotify_database.db`
- `spotify_data_V3.csv`
- `spotify_dataset.csv`
- `spotify_dataset_sample.csv`
- `data_with_famous_artist.csv`
- `spotify_dedup_by_day_title__with_combo_stats.csv`
- `data_clean_new.csv`

After downloading, place the files inside the project according to the directory tree above.  
If the `data/` directory already exists after cloning the repo, **replace its contents** with the downloaded version.



## Requirements

### Python Dependencies

```bash
# Core dependencies
python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

# Optional for enhanced functionality
scipy >= 1.7.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/VanceF-21/spotify_analysis_project.git
cd spotify_analysis_project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Data Preparation

#### All Command-Line Arguments

| Argument                                           | Type   | Default                                                                  | Description                                                                        |
| -------------------------------------------------- | ------ | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Data Configuration**                             |        |                                                                          |                                                                                    |
| `--input_raw_path`                                 | `str`  | `data/data_preparation/Spotify_Dataset_V3.csv`                           | Path to the raw Spotify Top-200 CSV file                                           |
| `--kmeans_map_path`                                | `str`  | `data/data_preparation/kmeans_clustered_data.csv`                        | Path to CSV with per-track K-means cluster assignments                             |
| `--output_clean_path`                              | `str`  | `data/data_preparation/spotify_cleaned_data.csv`                         | Path for the fully cleaned, deduplicated, feature-rich dataset                     |
| `--output_dir`                                     | `str`  | `data/data_preparation`                                                  | Base directory for data-preparation outputs                                        |
| **Cleaning Options (`spotify_data_cleaning.py`)**  |        |                                                                          |                                                                                    |
| `--deduplicate_by_day_title`                       | `bool` | `True`                                                                   | Whether to deduplicate tracks by `(Date, Title)` key                               |
| `--fix_loudness_outliers`                          | `bool` | `True`                                                                   | Correct extremely large-magnitude loudness values (divide by 1000)                 |
| `--clip_bounded_features`                          | `bool` | `True`                                                                   | Clip bounded audio features to `[0, 1]` (e.g. Danceability, Energy, Valence)       |
| `--compute_time_features`                          | `bool` | `True`                                                                   | Add calendar features (`year`, `month`, `week`, `weekday`, `is_weekend`, `yyyymm`) |
| `--compute_popularity_points`                      | `bool` | `True`                                                                   | Compute `rank_points`, `pop_points_total`, and `pop_points_artist`                 |
| `--merge_kmeans_clusters`                          | `bool` | `True`                                                                   | Merge external K-means cluster labels from `kmeans_clustered_data.csv`             |
| `--compute_combo_stats`                            | `bool` | `True`                                                                   | Compute per-artist-combo audio feature summaries (mean / variance / count)         |
| **Correlation Heatmap (`correlation_heatmap.py`)** |        |                                                                          |                                                                                    |
| `--heatmap_input_path`                             | `str`  | `data/data_preparation/data_clean_new.csv`                               | Input CSV used to compute the numeric feature correlation matrix                   |
| `--heatmap_output_dir`                             | `str`  | `data/data_preparation`                                                  | Directory for saving the correlation heatmap image                                 |
| `--heatmap_filename`                               | `str`  | `correlation_heatmap.png`                                                | Name of the exported heatmap (PNG)                                                 |
| **“Table 1” Summaries (`table1.py`)**              |        |                                                                          |                                                                                    |
| `--table_input_path`                               | `str`  | `data/data_preparation/spotify_dedup_by_day_title__with_combo_stats.csv` | Input CSV used for artist-level style summaries                                    |
| `--top_k_artists`                                  | `int`  | `8`                                                                      | Number of top artists (by `pop_points_artist`) to include in the summary           |
| `--top_k_features`                                 | `int`  | `3`                                                                      | Number of audio features to report per top artist                                  |
| **General Options**                                |        |                                                                          |                                                                                    |
| `--log_level`                                      | `str`  | `INFO`                                                                   | Logging verbosity. Choices: `DEBUG`, `INFO`, `WARN`, `ERROR`                       |
| `--overwrite`                                      | `bool` | `True`                                                                   | Overwrite existing output files if they already exist                              |

#### Example Commands

**Run cleaning and feature engineering pipeline:**
```bash
python scripts/data_preparation/spotify_data_cleaning.py \
    --input_raw_path data/data_preparation/Spotify_Dataset_V3.csv \
    --kmeans_map_path data/data_preparation/kmeans_clustered_data.csv \
    --output_clean_path data/data_preparation/spotify_cleaned_data.csv \
    --deduplicate_by_day_title True \
    --merge_kmeans_clusters True \
    --compute_combo_stats True
```

**Generate correlation heatmap from cleaned dataset:**
```bash
python scripts/data_preparation/correlation_heatmap.py \
    --heatmap_input_path data/data_preparation/data_clean_new.csv \
    --heatmap_output_dir data/data_preparation \
    --heatmap_filename correlation_heatmap.png
```

**Create “Table 1” artist summary tables:**
```bash
python scripts/data_preparation/table1.py \
    --table_input_path data/data_preparation/spotify_dedup_by_day_title__with_combo_stats.csv \
    --top_k_artists 8 \
    --top_k_features 3
```

**Expected Outputs:**
- Cleaned dataset with engineered features: spotify_cleaned_data.csv
- Correlation matrix heatmap image: correlation_heatmap.png
- Summary tables for report:
  overall_feature_means_across_all_artists.csv

  overall_feature_means_top3.csv, overall_feature_means_bottom3.csv

  top8_artists_by_pop_points.csv

  top8_artists_top3_single_features_wide.csv


### Task 1: Music Style Classification

#### All Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Data Configuration** |
| `--db_path` | `str` | `data/task1/spotify_database.db` | Path to SQLite database file |
| `--table_name` | `str` | `KmeanSample` | Table name in the database |
| `--output_dir` | `str` | `` | Base directory for output files |
| **Clustering Configuration** |
| `--n_clusters` | `int` | `8` | Number of clusters for K-means |
| `--random_state` | `int` | `42` | Random seed for reproducibility |
| `--clustering_method` | `str` | `kmeans++` | Clustering method to use. Choices: `kmeans`, `kmeans++`, `hierarchical` |
| **K-means Hyperparameters** |
| `--n_init` | `int` | `8` | Number of initializations for K-means |
| `--max_iter` | `int` | `200` | Maximum number of iterations for K-means |
| **Hierarchical Clustering Hyperparameters** |
| `--linkage` | `str` | `ward` | Linkage method for hierarchical clustering |
| `--sample_size` | `int` | `8693` | Sample size for hierarchical clustering visualization |
| **Feature Configuration** |
| `--features` | `str [...]` | All features | Features to use for clustering. Default: `Danceability` `Loudness` `Speechiness` `Acousticness` `Instrumentalness` `Valence` |
| **Analysis Options** |
| `--perform_stability_analysis` | `bool` | `False` | Whether to perform stability analysis |
| `--n_seeds` | `int` | `100` | Number of random seeds for stability analysis |
| **Performance** |
| `--n_jobs` | `int` | `-1` | Number of parallel jobs (-1 = all CPU cores) |

#### Example Commands

**Basic K-means++ clustering with 8 clusters:**
```bash
python scripts/music_style_classification/MAIN_Elbow_Point_Optimal_Clustering_k8.py \
    --db_path data/task1/spotify_database.db \
    --table_name KmeanSample \
    --n_clusters 8 \
    --random_state 42 \
    --output_dir  \
    --clustering_method kmeans++
```

**Run stability analysis for multiple K values:**
```bash
python scripts/music_style_classification/kmeans_stability_analysis.py \
    --db_path data/task1/spotify_database.db \
    --table_name KmeanSample \
    --output_dir results/task1/stability \
    --k_values 4 7 8 10 \
    --n_seeds 50 \
    --sample_size 8693 \
    --n_init 10 \
    --perform_stability_analysis True
```

**Feature and dimensionality analysis:**
```bash
python SPOTIFY_ANALYSIS_PROJECT/scripts/music_style_classification/SectionB\ Kmeans_feature_and_dimension.py \
    --db_path data/task1/spotify_database.db \
    --output_dir results/task1/dimension_analysis \
    --n_clusters 12 \
    --random_state 0
```

**Expected Outputs:**
- 8 distinct music style clusters (C0-C7)
- Cluster visualizations and statistics
- Cluster assignment for each track


### Task 2: Popularity Prediction

#### All Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Data Configuration** |
| `--data_path` | `str` | `data/task2/spotify_dataset.csv` | Path to dataset CSV file |
| `--output_dir` | `str` | `scripts/feature_popularity_analysis/results/cls` | Base directory for output files |
| **Model Configuration** |
| `--test_size` | `float` | `0.2` | Test set proportion (0-1) |
| `--random_state` | `int` | `42` | Random seed for reproducibility |
| `--models` | `str [...]` | All models | Models to test. Choices: `Decision Tree`, `Random Forest`, `Extra Trees`, `Gradient Boosting`, `HistGradient Boosting`, `AdaBoost` |
| **Random Forest Hyperparameters** |
| `--rf_n_estimators` | `int` | `100` | Number of trees in Random Forest |
| `--rf_max_depth` | `int` | `None` | Maximum depth of trees (None = unlimited) |
| **Gradient Boosting Hyperparameters** |
| `--gb_n_estimators` | `int` | `100` | Number of boosting stages |
| `--gb_max_depth` | `int` | `5` | Maximum depth of trees |
| `--gb_learning_rate` | `float` | `0.1` | Learning rate (shrinkage) |
| **AdaBoost Hyperparameters** |
| `--ada_n_estimators` | `int` | `50` | Number of estimators |
| **Performance** |
| `--n_jobs` | `int` | `-1` | Number of parallel jobs (-1 = all CPU cores) |

#### Example Commands

**Full experiment (custom hyperparameters):**
```bash
python scripts/feature_popularity_analysis/classification_binary.py \
    --data_path data/task2/spotify_dataset.csv \
    --output_dir scripts/feature_popularity_analysis/results/cls \
    --test_size 0.2 \
    --random_state 42 \
    --models "Decision Tree" "Random Forest" "Extra Trees" "Gradient Boosting" "HistGradient Boosting" "AdaBoost" \
    --rf_n_estimators 200 \
    --rf_max_depth 20 \
    --gb_n_estimators 150 \
    --gb_max_depth 7 \
    --gb_learning_rate 0.05 \
    --ada_n_estimators 100 \
    --n_jobs -1
```


#### Output Files

Each experiment creates a timestamped folder containing:

1. **feature_importance.pdf** - Top 20 feature importance visualization
2. **experiment_results.txt** - Comprehensive results including:
   - Model performance comparison (Accuracy, F1-score)
   - Best model classification report
   - Confusion matrix
   - Feature importance rankings
   - Cluster weight configuration


## Methodology

### Task 1: Clustering Approach

1. **Feature Selection:** Six de-correlated audio attributes (Danceability, Loudness, Speechiness, Acousticness, Instrumentalness, Valence)
2. **Preprocessing:** Standardization and redundancy reduction (removed Energy due to ρ=0.73 with Loudness)
3. **Algorithm:** K-Means++ with K=8, selected based on:
   - Silhouette score analysis
   - Elbow method
   - Stability (Adjusted Rand Index)
4. **Validation:** MDS plots and distance matrices confirm well-differentiated clusters

### Task 2: Classification Approach

1. **Feature Engineering:**
   - Weighted cluster features (C0-C7) derived from Task 1 centroids
   - Contextual variables: season, continent, nationality, is_weekend
   - Closest cluster assignment via Euclidean distance

2. **Target Variable:**
   - Binary classification: is_hit (1 = popular, 0 = less popular)
   - Median split ensures balanced dataset (50%-50%)

3. **Model Selection:**
   - Tree-based classifiers: Decision Tree, Random Forest, Extra Trees
   - Boosting methods: Gradient Boosting, HistGradient Boosting, AdaBoost
   - Evaluation metrics: Accuracy, F1-score, per-class precision/recall

4. **Pipeline:**
   - Numerical features: StandardScaler
   - Categorical features: One-hot encoding
   - Unified preprocessing via ColumnTransformer

## Key Results Summary

### Task 1: Music Style Identification

Eight distinct clusters identified:
- **C0:** Low-energy Emotional
- **C1:** High-energy Party (Energetic Pop)
- **C2:** Instrumental Beats
- **C3:** High-speech Rap
- **C4:** Acoustic Quiet
- **C5:** Powerful Electronic (Big-room Dance)
- **C6:** Mellow Blend
- **C7:** Positive Dance (Latin Pop)

### Task 2: Popularity Drivers

**Most Important Predictors:**
1. C4 (Acoustic) and C5 (Electronic) - Balance of niche and mainstream appeal
2. C2 (Instrumental) and C7 (Dance) - Stylistic distinction
3. Season and is_weekend - Temporal listening patterns

**Model Performance:**
- Best model: Random Forest (80.9% accuracy)
- Classification outperforms regression (threshold effects in popularity)
- Bagging methods superior to boosting on balanced data

## Research Insights

### Why Classification Over Regression?

Our experiments reveal three key advantages of binary classification:

1. **Threshold Effects:** Popularity exhibits discrete boundaries (viral vs. obscure) rather than smooth gradients
2. **Noise Mitigation:** Classification smooths irregular distributions by collapsing outcomes into robust decision boundaries
3. **Algorithmic Alignment:** Streaming success driven by gatekeeping and tipping points—inherently discrete phenomena

### Practical Applications

- **Playlist Curation:** Prioritize C5/C7 styles for mass appeal, C4/C2 for mood-driven playlists
- **Artist Development:** Guide production toward optimal style combinations
- **Promotional Strategy:** Time releases for high-activity periods (weekends, summer)
- **A&R Decisions:** Identify tracks balancing accessibility and stylistic distinction

## Future Work

1. **Temporal Validation:** Train on 2020-2025, test on 2026+ to assess stability
2. **Real-time Prediction:** Incorporate early streaming velocity and playlist inclusion rates
3. **Cross-platform Analysis:** Extend to Apple Music, YouTube Music for generalization
4. **Lyrical Features:** Integrate NLP analysis of lyrics with audio attributes
5. **Multi-label Classification:** Predict multiple success dimensions (chart longevity, playlist adds, social shares)

