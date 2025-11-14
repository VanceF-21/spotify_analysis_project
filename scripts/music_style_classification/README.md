# Task 1: Music Style Classification

## Overview

This task identifies eight representative music styles from Spotify's "Top 200" playlists using K-Means++ clustering on audio features. The resulting clusters serve as interpretable, high-level features for downstream popularity prediction.

## Method

- **Algorithm:** K-Means++ clustering with K=8
- **Features:** 6 audio attributes (Danceability, Loudness, Speechiness, Acousticness, Instrumentalness, Valence)
- **Preprocessing:** Standardization and redundancy reduction (removed Energy due to high correlation with Loudness)
- **Selection Criteria:** Silhouette score, Elbow method, and stability analysis (Adjusted Rand Index)

## Identified Music Styles

- **C0:** Balanced Mix
- **C1:** Energetic Pop
- **C2:** Instrumental Beats
- **C3:** Hip-hop
- **C4:** Acoustic
- **C5:** Powerful Electronic (Big-room Dance)
- **C6:** Mellow Blend
- **C7:** Positive Dance (Latin Pop)