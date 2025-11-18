import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
conn = sqlite3.connect(r"data/task1/spotify_database.db")
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
        style_name = " / ".join(dominant_feats[:2])
    
    style_labels.append((i, style_name))

style_dict = dict(style_labels)

# Define color palette
colors = ['#7EB5D6', '#D4A5A5', '#B4B4B4', '#D4B896', 
          '#8CB4A0', '#B4C7E7', '#C9C9C9', '#A5A5C8']

# ============================================================================
# FIGURE 1: ENHANCED DISTANCE HEATMAP
# ============================================================================
print("Generating enhanced distance heatmap...")
fig, ax = plt.subplots(figsize=(8, 7))

cluster_labels_short = [f"C{i}: {style_dict.get(i, '')}" for i in range(8)]
cmap = plt.cm.viridis_r

im = ax.imshow(distances, cmap=cmap, aspect='auto')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Euclidean Distance', rotation=270, labelpad=20, fontsize=11, weight='bold')
cbar.ax.tick_params(labelsize=9)

ax.set_xticks(np.arange(8))
ax.set_yticks(np.arange(8))
ax.set_xticklabels(cluster_labels_short, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(cluster_labels_short, fontsize=9)

ax.set_xticks(np.arange(8)-.5, minor=True)
ax.set_yticks(np.arange(8)-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

for i in range(8):
    for j in range(8):
        text_color = 'white' if distances[i, j] < distances.max()*0.5 else 'black'
        text = ax.text(j, i, f'{distances[i, j]:.2f}',
                      ha="center", va="center", color=text_color,
                      fontsize=8, weight='bold')

ax.set_title('Cluster Centroids Distance Matrix (K=8)', 
             fontsize=13, weight='bold', pad=15)

plt.tight_layout()
plt.savefig("enhanced_cluster_distance_heatmap_k8.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("enhanced_cluster_distance_heatmap_k8.pdf", bbox_inches='tight', facecolor='white')
print("Saved heatmap: enhanced_cluster_distance_heatmap_k8.png/pdf")
plt.close()

# ============================================================================
# FIGURE 2: 2D MDS VISUALIZATION
# ============================================================================
print("Generating 2D MDS visualization...")
mds_2d = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
mds_result_2d = mds_2d.fit_transform(distances)

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(8):
    ax.scatter(mds_result_2d[i, 0], mds_result_2d[i, 1], 
              s=800, alpha=0.7, c=colors[i], 
              edgecolors='white', linewidths=2.5,
              zorder=3)

closest_pairs = []
for i in range(len(distances)):
    for j in range(i+1, len(distances)):
        closest_pairs.append((i, j, distances[i, j]))

closest_pairs.sort(key=lambda x: x[2])

print("\nClosest cluster pairs:")
max_dist = closest_pairs[9][2]
min_dist = closest_pairs[0][2]

for idx, (c1, c2, dist) in enumerate(closest_pairs[:10]):
    print(f"{idx+1}. C{c1} - C{c2}: distance = {dist:.4f}")
    
    linewidth = 3.5 - 2.5 * (dist - min_dist) / (max_dist - min_dist)
    alpha = 0.6 - 0.35 * (dist - min_dist) / (max_dist - min_dist)
    
    ax.plot([mds_result_2d[c1, 0], mds_result_2d[c2, 0]],
           [mds_result_2d[c1, 1], mds_result_2d[c2, 1]],
           'k-', alpha=alpha, linewidth=linewidth, zorder=1)

legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='white', 
                                 linewidth=1.5, label=f'C{i}: {style_dict.get(i, "")}')
                  for i in range(8)]

legend = ax.legend(handles=legend_elements, 
                  loc='center left', bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fancybox=True, shadow=True,
                  fontsize=9, title='Cluster Styles', title_fontsize=10)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)

ax.set_xlabel('MDS Dimension 1', fontsize=11, weight='bold')
ax.set_ylabel('MDS Dimension 2', fontsize=11, weight='bold')
ax.set_title('2D MDS Visualization of Cluster Centroids (K=8)', 
            fontsize=13, weight='bold', pad=15)

ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.set_facecolor('#F8F8F8')
fig.patch.set_facecolor('white')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("enhanced_cluster_mds_2d_k8.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("enhanced_cluster_mds_2d_k8.pdf", bbox_inches='tight', facecolor='white')
print("Saved 2D MDS visualization: enhanced_cluster_mds_2d_k8.png/pdf")
plt.close()

# ============================================================================
# FIGURE 3: 3D MDS VISUALIZATION
# ============================================================================
print("Generating 3D MDS visualization...")
mds_3d = MDS(n_components=3, random_state=42, dissimilarity='precomputed')
mds_result_3d = mds_3d.fit_transform(distances)

# Create multiple views of 3D plot
fig = plt.figure(figsize=(14, 6))

# View angles for 2 subplots (top-left and bottom-right perspectives)
views = [(20, 300), (10, 225)]
subplot_positions = [121, 122]

for view_idx, (elev, azim) in enumerate(views):
    ax = fig.add_subplot(subplot_positions[view_idx], projection='3d')
    
    # Plot points
    for i in range(8):
        ax.scatter(mds_result_3d[i, 0], mds_result_3d[i, 1], mds_result_3d[i, 2],
                  s=600, alpha=0.8, c=colors[i], 
                  edgecolors='white', linewidths=2,
                  depthshade=True)
    
    # Draw connections for closest pairs
    for idx, (c1, c2, dist) in enumerate(closest_pairs[:10]):
        linewidth = 2.5 - 1.5 * (dist - min_dist) / (max_dist - min_dist)
        alpha = 0.4 - 0.2 * (dist - min_dist) / (max_dist - min_dist)
        
        ax.plot([mds_result_3d[c1, 0], mds_result_3d[c2, 0]],
               [mds_result_3d[c1, 1], mds_result_3d[c2, 1]],
               [mds_result_3d[c1, 2], mds_result_3d[c2, 2]],
               'k-', alpha=alpha, linewidth=linewidth)
    
    # Styling
    ax.set_xlabel('MDS Dim 1', fontsize=12, weight='bold', labelpad=8)
    ax.set_ylabel('MDS Dim 2', fontsize=12, weight='bold', labelpad=8)
    ax.set_zlabel('MDS Dim 3', fontsize=12, weight='bold', labelpad=8)
    
    ax.set_title(f'View: elev={elev}°, azim={azim}°', 
                fontsize=15, weight='bold', pad=15)
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor('#F8F8F8')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    
    # Set equal aspect ratio for all axes
    max_range = np.array([mds_result_3d[:, 0].max()-mds_result_3d[:, 0].min(),
                         mds_result_3d[:, 1].max()-mds_result_3d[:, 1].min(),
                         mds_result_3d[:, 2].max()-mds_result_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (mds_result_3d[:, 0].max()+mds_result_3d[:, 0].min()) * 0.5
    mid_y = (mds_result_3d[:, 1].max()+mds_result_3d[:, 1].min()) * 0.5
    mid_z = (mds_result_3d[:, 2].max()+mds_result_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add overall title and legend
fig.suptitle('3D MDS Visualization of Cluster Centroids (K=8)', 
            fontsize=20, weight='bold', y=0.98)

# Add legend at the bottom
legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='white', 
                                 linewidth=1.5, label=f'C{i}: {style_dict.get(i, "")}')
                  for i in range(8)]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
          frameon=True, fancybox=True, shadow=True,
          fontsize=9, title='Cluster Styles', title_fontsize=15,
          bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.subplots_adjust(wspace=0.15)  # Reduce horizontal space between subplots
plt.savefig("enhanced_cluster_mds_3d_k8.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("enhanced_cluster_mds_3d_k8.pdf", bbox_inches='tight', facecolor='white')
print("Saved 3D MDS visualization: enhanced_cluster_mds_3d_k8.png/pdf")
plt.close()

# ============================================================================
# BONUS: Single large 3D view for detailed examination
# ============================================================================
print("Generating single large 3D view...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot points with enhanced styling
for i in range(8):
    ax.scatter(mds_result_3d[i, 0], mds_result_3d[i, 1], mds_result_3d[i, 2],
              s=1000, alpha=0.8, c=colors[i], 
              edgecolors='white', linewidths=3,
              depthshade=True)

# Draw connections
for idx, (c1, c2, dist) in enumerate(closest_pairs[:10]):
    linewidth = 3 - 2 * (dist - min_dist) / (max_dist - min_dist)
    alpha = 0.5 - 0.3 * (dist - min_dist) / (max_dist - min_dist)
    
    ax.plot([mds_result_3d[c1, 0], mds_result_3d[c2, 0]],
           [mds_result_3d[c1, 1], mds_result_3d[c2, 1]],
           [mds_result_3d[c1, 2], mds_result_3d[c2, 2]],
           'k-', alpha=alpha, linewidth=linewidth)

ax.set_xlabel('MDS Dimension 1', fontsize=12, weight='bold', labelpad=10)
ax.set_ylabel('MDS Dimension 2', fontsize=12, weight='bold', labelpad=10)
ax.set_zlabel('MDS Dimension 3', fontsize=12, weight='bold', labelpad=10)
ax.set_title('3D MDS Visualization of Cluster Centroids (K=8)', 
            fontsize=14, weight='bold', pad=20)

ax.view_init(elev=25, azim=45)
ax.set_facecolor('#F8F8F8')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Equal aspect ratio
max_range = np.array([mds_result_3d[:, 0].max()-mds_result_3d[:, 0].min(),
                     mds_result_3d[:, 1].max()-mds_result_3d[:, 1].min(),
                     mds_result_3d[:, 2].max()-mds_result_3d[:, 2].min()]).max() / 2.0

mid_x = (mds_result_3d[:, 0].max()+mds_result_3d[:, 0].min()) * 0.5
mid_y = (mds_result_3d[:, 1].max()+mds_result_3d[:, 1].min()) * 0.5
mid_z = (mds_result_3d[:, 2].max()+mds_result_3d[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add legend
legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='white', 
                                 linewidth=1.5, label=f'C{i}: {style_dict.get(i, "")}')
                  for i in range(8)]
legend = ax.legend(handles=legend_elements, 
                  loc='center left', bbox_to_anchor=(1.05, 0.5),
                  frameon=True, fancybox=True, shadow=True,
                  fontsize=9, title='Cluster Styles', title_fontsize=10)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)

plt.tight_layout()
plt.savefig("enhanced_cluster_mds_3d_single_k8.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("enhanced_cluster_mds_3d_single_k8.pdf", bbox_inches='tight', facecolor='white')
print("Saved single 3D view: enhanced_cluster_mds_3d_single_k8.png/pdf")
plt.close()

# Save distance matrix
distance_df = pd.DataFrame(distances, 
                          index=[f"C{i}: {style_dict.get(i, '')}" for i in range(8)],
                          columns=[f"C{i}" for i in range(8)])
distance_df.to_csv("enhanced_cluster_distances_k8.csv", encoding='utf-8-sig')
print("\nSaved distance matrix: enhanced_cluster_distances_k8.csv")

print("\n" + "="*60)
print("Visualization complete!")
print("="*60)
print("\nGenerated files:")
print("1. enhanced_cluster_distance_heatmap_k8.png/pdf")
print("2. enhanced_cluster_mds_2d_k8.png/pdf")
print("3. enhanced_cluster_mds_3d_k8.png/pdf (2 views - compact)")
print("4. enhanced_cluster_mds_3d_single_k8.png/pdf (single detailed view)")
print("5. enhanced_cluster_distances_k8.csv")
