import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set publication-quality parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Load data
print("Loading data...")
conn = sqlite3.connect(r"spotify_data\spotify_database.db")
df = pd.read_sql("SELECT * FROM KmeanSample;", conn)
conn.close()

# Feature extraction
X = df[['Danceability','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']]

# Run KMeans++ clustering (K=8)
print("Running KMeans++ clustering...")
kmeans_plus = KMeans(n_clusters=8, init='k-means++', n_init=10,
                    max_iter=200, random_state=42)
labels_kmpp = kmeans_plus.fit_predict(X)

# Get cluster centroids
centroids = kmeans_plus.cluster_centers_
print(f"Centroids shape: {centroids.shape}")

# Calculate distance matrix between cluster centroids
print("Calculating distances between cluster centroids...")
distances = squareform(pdist(centroids, metric='euclidean'))

# Create style descriptions
df_clusters = X.copy()
df_clusters['cluster'] = labels_kmpp
cluster_summary = df_clusters.groupby('cluster').agg(['mean','std'])
global_mean = df_clusters.drop('cluster', axis=1).mean()

style_labels = []
for i in sorted(df_clusters['cluster'].unique()):
    dominant_feats = []
    for col in df_clusters.columns[:-1]:
        try:
            cluster_mean = cluster_summary.loc[i, (col, 'mean')]
            if cluster_mean > global_mean[col]:
                display_name = col.replace('_', ' ').title()
                dominant_feats.append(display_name)
        except KeyError:
            continue
    
    if not dominant_feats:
        style_name = "Neutral"
    else:
        style_name = " / ".join(dominant_feats[:2])  # Limit to top 2 features
    
    style_labels.append((i, style_name))

style_dict = dict(style_labels)

# Define color palette consistent with your existing figures
colors = ['#7EB5D6', '#D4A5A5', '#B4B4B4', '#D4B896', 
          '#8CB4A0', '#B4C7E7', '#C9C9C9', '#A5A5C8']

# ============================================================================
# FIGURE 1: ENHANCED DISTANCE HEATMAP
# ============================================================================
print("Generating enhanced distance heatmap...")
fig, ax = plt.subplots(figsize=(8, 7.5))

# Create labels with cluster ID and style
cluster_labels_short = [f"C{i}" for i in range(8)]

# Create custom colormap (inverted viridis for distance - darker = closer)
cmap = plt.cm.viridis_r

# Create heatmap with enhanced styling
im = ax.imshow(distances, cmap=cmap, aspect='auto')

# Add colorbar with better formatting
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Euclidean Distance', rotation=270, labelpad=20, fontsize=14, weight='bold')
cbar.ax.tick_params(labelsize=14)

# Set ticks and labels
ax.set_xticks(np.arange(8))
ax.set_yticks(np.arange(8))
ax.set_xticklabels(cluster_labels_short, rotation=45, ha='right', fontsize=14)
ax.set_yticklabels(cluster_labels_short, fontsize=14)

# Add grid
ax.set_xticks(np.arange(8)-.5, minor=True)
ax.set_yticks(np.arange(8)-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

# Add annotations with distance values
for i in range(8):
    for j in range(8):
        text_color = 'black' if distances[i, j] < distances.max()*0.5 else 'white'
        text = ax.text(j, i, f'{distances[i, j]:.2f}',
                      ha="center", va="center", color=text_color,
                      fontsize=12, weight='light')

# Title and layout
ax.set_title('Cluster Centroids Distance Matrix (K=8)', 
             fontsize=24, weight='bold', pad=20)

plt.tight_layout()
plt.savefig("enhanced_cluster_distance_heatmap_k8.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("enhanced_cluster_distance_heatmap_k8.pdf", bbox_inches='tight', facecolor='white')
print("Saved heatmap: enhanced_cluster_distance_heatmap_k8.png/pdf")

# ============================================================================
# FIGURE 2: ENHANCED MDS VISUALIZATION
# ============================================================================
print("Generating enhanced MDS visualization...")
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
mds_result = mds.fit_transform(distances)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot cluster points with enhanced styling
for i in range(8):
    # Main scatter point
    ax.scatter(mds_result[i, 0], mds_result[i, 1], 
              s=800, alpha=0.7, c=colors[i], 
              edgecolors='white', linewidths=2.5,
              zorder=3)
    
    # Add cluster label inside the point
    ax.text(mds_result[i, 0], mds_result[i, 1], f'C{i}',
           ha='center', va='center', fontsize=12, 
           weight='bold', color='white', zorder=4)

# Calculate and draw connections for closest pairs
closest_pairs = []
for i in range(len(distances)):
    for j in range(i+1, len(distances)):
        closest_pairs.append((i, j, distances[i, j]))

closest_pairs.sort(key=lambda x: x[2])

# Draw top 10 closest connections with varying thickness
print("\nClosest cluster pairs:")
max_dist = closest_pairs[9][2]  # 10th closest distance
min_dist = closest_pairs[0][2]  # closest distance

for idx, (c1, c2, dist) in enumerate(closest_pairs[:10]):
    print(f"{idx+1}. C{c1} - C{c2}: distance = {dist:.4f}")
    
    # Line thickness inversely proportional to distance
    linewidth = 3.5 - 2.5 * (dist - min_dist) / (max_dist - min_dist)
    alpha = 0.6 - 0.35 * (dist - min_dist) / (max_dist - min_dist)
    
    ax.plot([mds_result[c1, 0], mds_result[c2, 0]],
           [mds_result[c1, 1], mds_result[c2, 1]],
           'k-', alpha=alpha, linewidth=linewidth, zorder=1)

# Create legend with style descriptions
legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='white', 
                                 linewidth=1.5, label=f'C{i}: {style_dict.get(i, "")}')
                  for i in range(8)]

legend = ax.legend(handles=legend_elements, 
                  loc='center left', bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fancybox=True, shadow=True,
                  fontsize=9, title='Cluster Styles', title_fontsize=10)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)

# Styling
ax.set_xlabel('MDS Dimension 1', fontsize=11, weight='bold')
ax.set_ylabel('MDS Dimension 2', fontsize=11, weight='bold')
ax.set_title('MDS Visualization of Cluster Centroids (K=8)', 
            fontsize=13, weight='bold', pad=15)

# Add subtle grid
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# Set background color
ax.set_facecolor('#F8F8F8')
fig.patch.set_facecolor('white')

# Equal aspect ratio for proper distance representation
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("enhanced_cluster_mds_visualization_k8.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("enhanced_cluster_mds_visualization_k8.pdf", bbox_inches='tight', facecolor='white')
print("Saved MDS visualization: enhanced_cluster_mds_visualization_k8.png/pdf")

# ============================================================================
# Save distance matrix to CSV
# ============================================================================
distance_df = pd.DataFrame(distances, 
                          index=[f"C{i}: {style_dict.get(i, '')}" for i in range(8)],
                          columns=[f"C{i}" for i in range(8)])
distance_df.to_csv("enhanced_cluster_distances_k8.csv", encoding='utf-8-sig')
print("\nSaved distance matrix: enhanced_cluster_distances_k8.csv")

print("\nVisualization complete!")
print("\nGenerated files:")
print("1. enhanced_cluster_distance_heatmap_k8.png/pdf")
print("2. enhanced_cluster_mds_visualization_k8.png/pdf")
print("3. enhanced_cluster_distances_k8.csv")
