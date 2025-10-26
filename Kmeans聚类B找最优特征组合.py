import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix

# 2. 连接 / 创建同文件夹下的 database.sqlite
conn = sqlite3.connect(r"C:\Users\LENOVO\Desktop\硕士学习\AML\Mini-Project\spotify_data\spotify_database.db")
df_train = pd.read_sql("SELECT * FROM KmeanSample2021Train;", conn)
X_train = df_train[['Danceability','Energy','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']]

df_val = pd.read_sql("SELECT * FROM KmeanSample2022Val;", conn)
X_val = df_val[['Danceability','Energy','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']]

feats = ['Danceability','Energy','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']


# S0: 无PCA
pipe_s0 = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=12, init="k-means++", n_init=50, random_state=0))
])

# S1: 删一列（假设 Energy 是那对高度相关里更弱的一列）
features_keep = [c for c in feats if c != "Energy"]
X_train_s1 = X_train[features_keep]
X_val_s1   = X_val[features_keep]
pipe_s1 = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=12, init="k-means++", n_init=50, random_state=0))
])

# S2: PCA-whitening
pipe_s2 = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(whiten=True, n_components=0.9, svd_solver="full")),
    ("kmeans", KMeans(n_clusters=12, init="k-means++", n_init=50, random_state=0))
])

# 仅在 train(≤2023) 拟合；val/test 只 transform/predict
pipe_s0.fit(X_train); labels_s0 = pipe_s0.predict(X_val)
pipe_s1.fit(X_train_s1); labels_s1 = pipe_s1.predict(X_val_s1)
pipe_s2.fit(X_train); labels_s2 = pipe_s2.predict(X_val)




# 1) 看前几条预测
print("S0 head:", labels_s0[:10])
print("S1 head:", labels_s1[:10])
print("S2 head:", labels_s2[:10])

# 2) 各簇样本数
print("S0 counts:\n", pd.Series(labels_s0).value_counts().sort_index())
print("S1 counts:\n", pd.Series(labels_s1).value_counts().sort_index())
print("S2 counts:\n", pd.Series(labels_s2).value_counts().sort_index())

# 3) 训练集的 KMeans 惯性（SSE）
print("S0 train inertia:", pipe_s0.named_steps["kmeans"].inertia_)
print("S1 train inertia:", pipe_s1.named_steps["kmeans"].inertia_)
print("S2 train inertia:", pipe_s2.named_steps["kmeans"].inertia_)

# 4) 把簇标签并回验证集，便于后续分析/导出
df_val_out = df_val.copy()
df_val_out["cluster_s0"] = labels_s0
df_val_out["cluster_s1"] = labels_s1
df_val_out["cluster_s2"] = labels_s2


import numpy as np
def val_sse(pipe, X):
    km = pipe.named_steps["kmeans"]
    dists = km.transform(pipe[:-1].transform(X)).min(axis=1)  # 最近中心距离
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

# Jaccard（基于成对同簇关系）
print("Jaccard S0 vs S1:", clustering_jaccard(labels_s0, labels_s1))
print("Jaccard S0 vs S2:", clustering_jaccard(labels_s0, labels_s2))
print("Jaccard S1 vs S2:", clustering_jaccard(labels_s1, labels_s2))
