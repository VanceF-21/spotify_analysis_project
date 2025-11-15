import pandas as pd
import sqlite3
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix

# 确保results文件夹存在
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Connect to / create database.sqlite in the same folder
conn = sqlite3.connect(r"spotify_data\spotify_database.db")
df_train = pd.read_sql("SELECT * FROM KmeanSample2021Train;", conn)
X_train = df_train[['Danceability','Energy','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']]

df_val = pd.read_sql("SELECT * FROM KmeanSample2022Val;", conn)
X_val = df_val[['Danceability','Energy','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']]

feats = ['Danceability','Energy','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']


# S0: No PCA
pipe_s0 = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=12, init="k-means++", n_init=50, random_state=0))
])

# S1: Drop one column (assume Energy is the weaker variable in a highly correlated pair)
features_keep = [c for c in feats if c != "Energy"]
X_train_s1 = X_train[features_keep]
X_val_s1   = X_val[features_keep]
pipe_s1 = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=12, init="k-means++", n_init=50, random_state=0))
])

# S2: PCA whitening
pipe_s2 = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(whiten=True, n_components=0.9, svd_solver="full")),
    ("kmeans", KMeans(n_clusters=12, init="k-means++", n_init=50, random_state=0))
])

# Fit only on train (≤2023); val/test only transform/predict
pipe_s0.fit(X_train); labels_s0 = pipe_s0.predict(X_val)
pipe_s1.fit(X_train_s1); labels_s1 = pipe_s1.predict(X_val_s1)
pipe_s2.fit(X_train); labels_s2 = pipe_s2.predict(X_val)


# 1) Show first few predictions
print("S0 head:", labels_s0[:10])
print("S1 head:", labels_s1[:10])
print("S2 head:", labels_s2[:10])

# 2) Cluster sample sizes
print("S0 counts:\n", pd.Series(labels_s0).value_counts().sort_index())
print("S1 counts:\n", pd.Series(labels_s1).value_counts().sort_index())
print("S2 counts:\n", pd.Series(labels_s2).value_counts().sort_index())

# 3) KMeans inertia (SSE) on the training set
print("S0 train inertia:", pipe_s0.named_steps["kmeans"].inertia_)
print("S1 train inertia:", pipe_s1.named_steps["kmeans"].inertia_)
print("S2 train inertia:", pipe_s2.named_steps["kmeans"].inertia_)

# 4) Attach cluster labels back to the validation set for later analysis/export
df_val_out = df_val.copy()
df_val_out["cluster_s0"] = labels_s0
df_val_out["cluster_s1"] = labels_s1
df_val_out["cluster_s2"] = labels_s2

# 保存带聚类标签的数据到results文件夹
df_val_out.to_csv(os.path.join(output_dir, 'clustered_validation_data.csv'), index=False, encoding='utf-8-sig')
print(f"Clustered validation data saved to: {os.path.join(output_dir, 'clustered_validation_data.csv')}")


import numpy as np
def val_sse(pipe, X):
    km = pipe.named_steps["kmeans"]
    dists = km.transform(pipe[:-1].transform(X)).min(axis=1)  # Distance to the nearest cluster center
    return np.square(dists).sum()

print("S0 val SSE:", val_sse(pipe_s0, X_val))
print("S1 val SSE:", val_sse(pipe_s1, X_val_s1))
print("S2 val SSE:", val_sse(pipe_s2, X_val))


def clustering_jaccard(labels_a, labels_b):
    tn, fp, fn, tp = pair_confusion_matrix(labels_a, labels_b).ravel()
    return tp / (tp + fp + fn)

# ARI
print("ARI S0 vs S1:", adjusted_rand_score(labels_s0, labels_s1))
print("ARI S0 vs S2:", adjusted_rand_score(labels_s0, labels_s2))
print("ARI S1 vs S2:", adjusted_rand_score(labels_s1, labels_s2))

# Jaccard (based on pairwise co-clustering relationships)
print("Jaccard S0 vs S1:", clustering_jaccard(labels_s0, labels_s1))
print("Jaccard S0 vs S2:", clustering_jaccard(labels_s0, labels_s2))
print("Jaccard S1 vs S2:", clustering_jaccard(labels_s1, labels_s2))

# 保存所有性能指标到results文件夹
metrics_filename = os.path.join(output_dir, 'clustering_performance_metrics.txt')
with open(metrics_filename, 'w', encoding='utf-8') as f:
    f.write("KMEANS FEATURE AND DIMENSIONALITY REDUCTION ANALYSIS RESULTS\n")
    f.write("==================================================\n\n")
    
    # 1) Cluster sizes
    f.write("CLUSTER SIZES:\n")
    f.write("S0 counts:\n")
    f.write(f"{pd.Series(labels_s0).value_counts().sort_index()}\n\n")
    f.write("S1 counts:\n")
    f.write(f"{pd.Series(labels_s1).value_counts().sort_index()}\n\n")
    f.write("S2 counts:\n")
    f.write(f"{pd.Series(labels_s2).value_counts().sort_index()}\n\n")
    
    # 2) Inertia values
    f.write("INERTIA VALUES (TRAINING SET):\n")
    f.write(f"S0 train inertia: {pipe_s0.named_steps['kmeans'].inertia_}\n")
    f.write(f"S1 train inertia: {pipe_s1.named_steps['kmeans'].inertia_}\n")
    f.write(f"S2 train inertia: {pipe_s2.named_steps['kmeans'].inertia_}\n\n")
    
    # 3) SSE values
    f.write("SSE VALUES (VALIDATION SET):\n")
    f.write(f"S0 val SSE: {val_sse(pipe_s0, X_val)}\n")
    f.write(f"S1 val SSE: {val_sse(pipe_s1, X_val_s1)}\n")
    f.write(f"S2 val SSE: {val_sse(pipe_s2, X_val)}\n\n")
    
    # 4) ARI values
    f.write("ADJUSTED RAND INDEX (ARI):\n")
    f.write(f"ARI S0 vs S1: {adjusted_rand_score(labels_s0, labels_s1)}\n")
    f.write(f"ARI S0 vs S2: {adjusted_rand_score(labels_s0, labels_s2)}\n")
    f.write(f"ARI S1 vs S2: {adjusted_rand_score(labels_s1, labels_s2)}\n\n")
    
    # 5) Jaccard values
    f.write("JACCARD INDEX:\n")
    f.write(f"Jaccard S0 vs S1: {clustering_jaccard(labels_s0, labels_s1)}\n")
    f.write(f"Jaccard S0 vs S2: {clustering_jaccard(labels_s0, labels_s2)}\n")
    f.write(f"Jaccard S1 vs S2: {clustering_jaccard(labels_s1, labels_s2)}\n")

print(f"Performance metrics saved to: {metrics_filename}")
