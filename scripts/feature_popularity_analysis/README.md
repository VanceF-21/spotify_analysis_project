# Task 2: Popularity Prediction

## Overview

This task predicts song popularity using cluster-based features from Task 1 combined with contextual variables. Binary classification models achieve over 80% accuracy by leveraging style representations and temporal listening patterns.

## Method

- **Input Features:** 8 weighted cluster features (C0-C7) + contextual variables (season, continent, nationality, weekend)
- **Target Variable:** Binary classification (popular vs. less popular) using median split for balanced dataset
- **Models:** Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradient Boosting, AdaBoost
- **Evaluation:** Accuracy, F1-score, confusion matrix, feature importance analysis