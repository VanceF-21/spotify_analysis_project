import numpy as np 
import pandas as pd
import sqlite3
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, pair_confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
# C1 Baseline Clustering: K-Means (Random Initialization)

conn = sqlite3.connect("data/task1/spotify_database.db")
df = pd.read_sql("SELECT * FROM KmeanSample;", conn)

X = df[['Danceability','Loudness','Speechiness',
         'Acousticness','Instrumentalness','Valence']]

# Pre-sampling for silhouette score calculation to improve performance
sample_size = min(10000, len(X))  # Max 5000 samples
print(f"Using {sample_size} samples for silhouette score calculation to improve performance...")
sample_idx = np.random.choice(range(len(X)), size=sample_size, replace=False)
X_sample_for_silhouette = X.iloc[sample_idx]

# C1 Standard KMeans (Random Initialization) - Optimized Parameters
print("Starting C1 standard KMeans...")
kmeans_random = KMeans(n_clusters=8, init='random', n_init=8, 
                      max_iter=200, random_state=42)
labels_km = kmeans_random.fit_predict(X)
inertia_km = kmeans_random.inertia_
print("C1 Baseline Clustering Inertia:", inertia_km)

# Compute silhouette score using sampled data to improve performance
print("Calculating C1 silhouette score...")
sample_labels_km = kmeans_random.predict(X_sample_for_silhouette)
sil_km = silhouette_score(X_sample_for_silhouette, sample_labels_km)

# C2 Improved Clustering: K-Means++ (Default Initialization) - Optimized Parameters
print("Starting C2 K-Means++...")
kmeans_plus = KMeans(n_clusters=8, init='k-means++', n_init=8,
                    max_iter=200, random_state=42)
labels_kmpp = kmeans_plus.fit_predict(X)
inertia_kmpp = kmeans_plus.inertia_
print("C2 Improved Clustering Inertia:", inertia_kmpp)

# Compute silhouette score using sampled data to improve performance
print("Calculating C2 silhouette score...")
sample_labels_kmpp = kmeans_plus.predict(X_sample_for_silhouette)
sil_kmpp = silhouette_score(X_sample_for_silhouette, sample_labels_kmpp)

# Compare clustering consistency between the two approaches
ari_kmeans = adjusted_rand_score(labels_km, labels_kmpp)
print("C1 vs C2 Clustering Consistency ARI:", ari_kmeans)

# C3 Hierarchical Clustering
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# ---------- Parameters ----------
# ================== Heatmap (left, with wide left margin) + Dendrogram (right) ==================
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import textwrap

# -------- Params --------
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.patches import Rectangle
import textwrap

# -------- Params --------
K_TARGET = 8
SAMPLE_N = 8693
LINKAGE   = 'ward'
RANDOM_STATE = 42

SHOW_LEAF_LABELS = False         # Whether to show C1..CK labels at the bottom (usually False for cleaner look)
WRAP_LABELS = False              # Whether to wrap long row labels on right figure
WRAP_WIDTH = 48                  # Wrap width (# of characters), effective only when WRAP_LABELS=True

# Layout margins / formatting
RIGHT_MARGIN_FRACTION = 0.20     # Right margin fraction (0–1); increase if row labels are long
LEFT_MARGIN_FRACTION  = 0.12     # Left margin (for left plot), usually not very large
BOTTOM_MARGIN_FRACTION= 0.30     # Bottom margin; increase when feature names are long
XLABEL_ROTATION = 40             # X-axis label rotation for heatmap
XLABEL_PAD = 6                   # Pixel padding between x-axis labels and axis

# (Optional) Custom color palette: used for left dendrogram (right figure automatically syncs colors)
cluster_palette = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]

# -------- Input --------
X_np = X.values if hasattr(X, "values") else np.asarray(X)

# Standardization (strongly recommended for Ward/Euclidean distance)
X_std = StandardScaler().fit_transform(X_np)

# Sampling (without replacement)
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_std), size=min(SAMPLE_N, len(X_std)), replace=False)
X_sample = X_std[idx, :]

# Hierarchical clustering
Z = linkage(X_sample, method=LINKAGE)
labels = fcluster(Z, t=K_TARGET, criterion='maxclust')

# Slightly reduce value at the (K-1)-th merge to ensure cutting into K clusters
color_thr = Z[-(K_TARGET-1), 2] - 1e-9

# ---------- Cluster centers (relative to global mean) + row labels ----------
feature_names = getattr(X, "columns", [f"feat{i+1}" for i in range(X_sample.shape[1])])
global_mean = X_sample.mean(axis=0)

centroids = np.zeros((K_TARGET, X_sample.shape[1]))
for c in range(1, K_TARGET+1):
    m = (labels == c)
    centroids[c-1] = X_sample[m].mean(axis=0) if m.any() else global_mean
centroids_diff = centroids - global_mean

def top_k_feats(row, names, k=2):
    idx2 = np.argsort(row)[::-1][:k]
    return " / ".join([names[i] for i in idx2])

cluster_tags = [top_k_feats(centroids_diff[i], feature_names, k=2) for i in range(K_TARGET)]

# -------- Layout --------
plt.style.use('seaborn-v0_8-white')

# Increase figure size and DPI
fig = plt.figure(figsize=(16, 7.5), dpi=200)
# Left: dendrogram; Right: heatmap. Increase wspace for colorbar between the two.
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.5, 1.35], wspace=0.24)

# ================= Left: Dendrogram (compact + colored) =================
ax1 = fig.add_subplot(gs[0, 0])

# Set branch colors for left dendrogram (right figure auto-syncs colors)
set_link_color_palette(cluster_palette)

# Compact view of upper tree levels
dendrogram(
    Z,
    truncate_mode='level',    # Show only top hierarchical levels
    p=5,                      # 4 levels enough; can use 5 for more detail
    color_threshold=color_thr,
    show_contracted=False,
    no_labels=not SHOW_LEAF_LABELS,
    leaf_font_size=9,
    above_threshold_color="#9e9e9e",
    ax=ax1
)

# Cut line
ax1.axhline(y=color_thr, linestyle='--', linewidth=1.2, color='black')

# Enhance line clarity
for l in ax1.get_lines():
    l.set_linewidth(1.4)
    l.set_solid_capstyle('round')

# Bottom labels (optional)
if SHOW_LEAF_LABELS:
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels([f"C{i+1}" for i in range(K_TARGET)], rotation=0, fontsize=15)
else:
    ax1.set_xticks([])

ax1.set_xlabel("Merged clusters (top levels)", fontsize=15,labelpad=18)
ax1.set_ylabel("Distance", fontsize=15)

# ================= Key: Extract order & colors from complete tree (to sync with left plot) =================
d_full = dendrogram(Z, no_plot=True, color_threshold=color_thr)
leaf_order = d_full['leaves']                 # sample indices from left to right
leaf_colors = d_full['leaves_color_list']     # same order
sample_color = {leaf_order[i]: leaf_colors[i] for i in range(len(leaf_order))}

# Determine cluster display order by "first appearance" from left to right
seen, order_list, repr_sample_idx = set(), [], {}
for i_samp in leaf_order:
    k = int(labels[i_samp]) - 1
    if k not in seen:
        seen.add(k)
        order_list.append(k)
        repr_sample_idx[k] = i_samp
    if len(order_list) == K_TARGET:
        break

# Rare fallback if some cluster not shown
if len(order_list) < K_TARGET:
    for k in range(K_TARGET):
        if k not in seen:
            idxs = np.where(labels == (k+1))[0]
            repr_sample_idx[k] = int(idxs[0]) if len(idxs) else 0
            order_list.append(k)

order = np.array(order_list, dtype=int)
assert order.min() >= 0 and order.max() < K_TARGET and len(order) == K_TARGET, f"Bad order: {order}"

# Reorder right-figure rows & labels and assign cluster colors
centroids_diff_ord = centroids_diff[order]
row_labels = [f"C{idx+1}" for idx in order]
if WRAP_LABELS:
    row_labels = [textwrap.fill(lbl, width=WRAP_WIDTH, break_long_words=False, break_on_hyphens=False)
                  for lbl in row_labels]
row_colors = [sample_color[repr_sample_idx[idx]] for idx in order]  # synced with left figure

# ================= Right: Heatmap (row labels on the right; colorbar on the left) =================
ax2 = fig.add_subplot(gs[0, 1])

# Symmetric color scale with light clipping of extremes
v_abs = np.percentile(np.abs(centroids_diff_ord), 98)
im = ax2.imshow(centroids_diff_ord, aspect='auto', cmap='RdBu_r',
                vmin=-v_abs, vmax=v_abs, interpolation='nearest')

# Y-axis labels (right side)
ax2.set_yticks(np.arange(K_TARGET))
ax2.set_yticklabels(row_labels, fontsize=12)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', pad=6)

# Color blocks beside row labels
swatch_ax = ax2.inset_axes([0.992, 0.0, 0.014, 1.0], transform=ax2.transAxes)
swatch_ax.set_xlim(0, 1)
swatch_ax.set_ylim(-0.5, K_TARGET-0.5)
for r in range(K_TARGET):
    swatch_ax.add_patch(Rectangle((0, r-0.5), 1, 1, facecolor=row_colors[r], edgecolor='none'))
swatch_ax.axis('off')

# X-axis
ax2.set_xticks(np.arange(len(feature_names)))
ax2.set_xticklabels(feature_names, rotation=XLABEL_ROTATION, ha='right', fontsize=12)
ax2.tick_params(axis='x', pad=XLABEL_PAD)



# Place the colorbar on the "left side" of the heatmap (between the two plots) to achieve visual balance (Matplotlib >= 3.6)
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.08, location='left')
cbar.set_label("lower  ←  0  →  higher", rotation=90, fontsize=12)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
footer_ax = fig.add_axes([0.08, 0.03, 0.87, 0.2])  # NEW: occupy about 22% of the height at the bottom
footer_ax.axis('off')

# Put the summary description in the center of the footer (can be lower, see tunable parameters below)
summary_text = (
    f"Left is hierarchical clustering result (sample n={len(X_sample)}, K={K_TARGET}, linkage={LINKAGE}); "
    f"Right is cluster centroids (standardized mean − global mean)"
)
footer_ax.text(0.5, 0.5, summary_text, fontsize=15, color='black',
               ha='center', va='center', wrap=True)  # NEW

# -------- Margins (reserve blank space on right and bottom; left usually smaller) --------
plt.subplots_adjust(left=LEFT_MARGIN_FRACTION,
                    right=1 - RIGHT_MARGIN_FRACTION,
                    bottom=BOTTOM_MARGIN_FRACTION)

# Save as high-resolution vector graphic in SVG format
svg_filename = 'scripts/music_style_classification/results/hierarchical_clustering_heatmap_k=8.svg'
plt.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
print(f"Saved high-resolution vector image to: {svg_filename}")

# Also save as high-DPI PNG as a backup
png_filename = 'scripts/music_style_classification/results/hierarchical_clustering_heatmap_k=8.png'
plt.savefig(png_filename, format='png', dpi=300, bbox_inches='tight')
print(f"Saved high-resolution PNG image to: {png_filename}")

plt.show()

# -------- Results Summary --------
try:
    sizes = dict(sorted(Counter(labels).items()))
    print(f"Cluster sizes (on sample, K={K_TARGET}): {sizes}")
    sil_hc = silhouette_score(X_sample, labels)
    print(f"Silhouette (sample, standardized) = {sil_hc:.3f}")
    print("Heatmap row order (top->bottom):", [f"C{idx+1}" for idx in order])
    
    # Compute inertia of hierarchical clustering
    # Inertia is the sum of squared Euclidean distances from samples to their cluster centers
    inertia_hc = 0
    for i in range(len(X_sample)):
        cluster_idx = int(labels[i]) - 1  # Convert to 0-based index
        # Use non-reordered centroids
        cluster_center = centroids[cluster_idx]
        # Compute squared Euclidean distance
        distance_squared = np.sum((X_sample[i] - cluster_center) ** 2)
        inertia_hc += distance_squared
    print(f"Hierarchical Clustering Inertia (standardized sample): {inertia_hc:.3f}")
    
    # Unified output of key metrics for all clustering methods
    print("\n=== Summary of Key Metrics for All Clustering Methods ===")
    print("{:<25} {:<15} {:<15}".format("Clustering Method", "Inertia", "Silhouette Score"))
    print("-" * 55)
    print("{:<25} {:<15.3f} {:<15.3f}".format("Standard KMeans (Random)", inertia_km, sil_km))
    print("{:<25} {:<15.3f} {:<15.3f}".format("K-Means++", inertia_kmpp, sil_kmpp))
    print("{:<25} {:<15.3f} {:<15.3f}".format("Hierarchical (Std)", inertia_hc, sil_hc))
    
    # Find the best silhouette score
    best_method = max([('Standard KMeans', sil_km), ('K-Means++', sil_kmpp), ('Hierarchical Clustering', sil_hc)],
                      key=lambda x: x[1])
    print(f"\nBest Silhouette Score Method: {best_method[0]} ({best_method[1]:.3f})")
    
    # Find the minimum inertia (among all methods)
    min_inertia_val = min(inertia_km, inertia_kmpp, inertia_hc)
    if inertia_km == min_inertia_val:
        best_inertia = "Standard KMeans (Random Initialization)"
    elif inertia_kmpp == min_inertia_val:
        best_inertia = "K-Means++"
    else:
        best_inertia = "Hierarchical Clustering"
    print(f"Minimum Inertia Method: {best_inertia} ({min_inertia_val:.3f})")
    
except Exception as e:
    print("Summary not computed:", e)

# C5 Cluster Naming and Interpretation
# Keep original column names consistent, avoid case and spelling errors
df_clusters = X.copy()
df_clusters['cluster'] = labels_kmpp  # Use KMeans++ clustering result

print("\n=== C5 Cluster Naming and Interpretation ===")
print("Original data column names:", X.columns.tolist())
print("Number of cluster labels:", len(df_clusters['cluster'].unique()))

cluster_summary = df_clusters.groupby('cluster').agg(['mean','std'])
print("\nFeature statistics for each cluster:")
print("Number of clusters:", len(cluster_summary))
print("Feature statistics dimensions:", cluster_summary.shape)

# Show statistics of the first few clusters to avoid overly long output
if not cluster_summary.empty:
    print("\nFeature statistics for first ten clusters:")
    first_clusters = len(cluster_summary)
    for i in range(first_clusters):
        print(f"\nCluster {i}:")
        for col in cluster_summary.columns.levels[0][:10]:  # Only show the first few features
            mean_val = cluster_summary.loc[i, (col, 'mean')]
            std_val = cluster_summary.loc[i, (col, 'std')]
            print(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
else:
    print("Warning: No valid cluster statistics generated!")

# Automatically generate style names (features higher than global mean will be listed)
global_mean = df_clusters.drop('cluster', axis=1).mean()
style_labels = []
for i in sorted(df_clusters['cluster'].unique()):
    dominant_feats = []
    for col in df_clusters.columns[:-1]:  # Exclude 'cluster' column
        try:
            # Try to get the mean of this feature within the cluster
            cluster_mean = cluster_summary.loc[i, (col, 'mean')]
            if cluster_mean > global_mean[col]:
                # Optionally format feature names to be more readable
                display_name = col.replace('_', ' ').title()
                dominant_feats.append(display_name)
        except KeyError:
            # Handle potential column name mismatch
            continue
    
    # If no dominant features, mark as "Neutral"
    if not dominant_feats:
        style_name = "Neutral"
    else:
        style_name = " / ".join(dominant_feats)
    
    style_labels.append((i, style_name))

style_df = pd.DataFrame(style_labels, columns=['Cluster ID', 'Style Description'])
print("\nC5 Cluster Naming Results:\n", style_df)

# C6 Output final clustered result DataFrame

def get_final_clustered_dataframe():
    """
    Get and return the final clustered result DataFrame.
    Includes original data, cluster labels, and automatically generated style descriptions.
    
    Returns:
        pd.DataFrame: Complete DataFrame containing clustering results.
    """
    print("\n=== C6 Generate Final Clustered Result DataFrame ===")
    
    # Use KMeans++ clustering results
    print("Using KMeans++ clustering method results...")
    
    # Create DataFrame with original data and cluster labels
    final_df = df.copy()  # Use original DataFrame
    final_df['cluster_id'] = labels_kmpp  # Add cluster ID
    final_df['cluster_label'] = f"C" + final_df['cluster_id'].astype(str)  # Add cluster label (e.g., C0, C1, etc.)
    
    # Add style description
    style_dict = dict(style_labels)
    final_df['style_description'] = final_df['cluster_id'].map(style_dict)
    
    # Show result information
    print(f"Shape of final clustered result DataFrame: {final_df.shape}")
    print(f"Number of clusters: {len(final_df['cluster_id'].unique())}")
    print(f"Preview of first 5 rows:")
    print(final_df.head())
    
    # Save result to CSV file (optional)
    csv_filename = "scripts/music_style_classification/results/kmeans_clustered_data_deduplicated.csv"
    final_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\nSaved clustered results to CSV file: {csv_filename}")
    
    # Save result to SQLite database (optional)
    try:
        conn = sqlite3.connect(r"data/task1/spotify_database.db")
        final_df.to_sql("KmeansClusteredResults", conn, if_exists='replace', index=False)
        print(f"Saved clustered results to database table: KmeansClusteredResults")
        conn.close()
    except Exception as e:
        print(f"Error while saving to database: {e}")
    
    return final_df

# Generate and output the final clustered result DataFrame
final_clustered_df = get_final_clustered_dataframe()

# Show detailed statistics for each cluster
print("\n=== Detailed Statistics for Each Cluster ===")

# Compute total number of samples
total_samples = len(final_clustered_df)
print(f"Total number of samples: {total_samples}")

# Re-create style_dict
style_dict = dict(style_labels)

# Get sample count per cluster and sort
cluster_counts = final_clustered_df['cluster_id'].value_counts().sort_index()

# Compute cluster percentage
cluster_percentages = {cluster_id: (count / total_samples) * 100 
                      for cluster_id, count in cluster_counts.items()}

# Sort by sample count in descending order to find largest and smallest clusters
sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)

print("\n=== Sorted by Sample Count (Descending) ===")
print("{:<10} {:<15} {:<40} {:<15}".format("Cluster ID", "Song Count", "Style Description", "Percentage(%)"))
print("-" * 85)

for cluster_id, count in sorted_clusters:
    style_desc = style_dict.get(cluster_id, "Unknown")
    percentage = cluster_percentages[cluster_id]
    print("{:<10} {:<15} {:<40} {:<15.2f}".format(cluster_id, count, style_desc, percentage))

# Statistical summary
largest_cluster_id, largest_count = sorted_clusters[0]
smallest_cluster_id, smallest_count = sorted_clusters[-1]

print("\n=== Statistical Summary ===")
print(f"Largest cluster: Cluster {largest_cluster_id} ({style_dict.get(largest_cluster_id, 'Unknown')}) - {largest_count} songs ({cluster_percentages[largest_cluster_id]:.2f}%)")
print(f"Smallest cluster: Cluster {smallest_cluster_id} ({style_dict.get(smallest_cluster_id, 'Unknown')}) - {smallest_count} songs ({cluster_percentages[smallest_cluster_id]:.2f}%)")
print(f"Cluster size difference ratio: {largest_count / smallest_count:.2f}x")
print(f"Average cluster size: {total_samples / len(cluster_counts):.2f} songs")

# Generate pie chart visualization (optional)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    labels = [f"Cluster {cluster_id}\n{style_dict.get(cluster_id, '')}\n{count} songs ({cluster_percentages[cluster_id]:.1f}%)" 
              for cluster_id, count in sorted_clusters]
    sizes = [count for _, count in sorted_clusters]
    explode = [0.05] * len(sizes)  # Slightly separate all slices to enhance readability
    
    plt.pie(sizes, explode=explode, labels=labels, autopct=None, 
            startangle=90, textprops={'fontsize': 9}, shadow=False)
    plt.axis('equal')  # Ensure the pie chart is a circle
    plt.title('Distribution of Song Count by Cluster', fontsize=14, pad=20)
    
    # Save pie chart
    pie_filename = 'scripts/music_style_classification/results/cluster_distribution_pie_chart_k=8.png'
    plt.savefig(pie_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved cluster distribution pie chart to: {pie_filename}")
except Exception as e:
    print(f"\nError generating visualization chart: {e}")

# Add new functionality: normalize features to [0,1] and compute mean and std for each cluster
def calculate_normalized_cluster_stats(original_df, feature_columns, cluster_labels, cluster_ids):
    """
    Normalize features to the [0,1] range and compute mean and standard deviation for each cluster.
    
    Args:
        original_df: Original DataFrame.
        feature_columns: List of feature column names to normalize.
        cluster_labels: Array of cluster labels.
        cluster_ids: List of cluster IDs.
    
    Returns:
        pd.DataFrame: Result containing normalized feature means and standard deviations for each cluster.
    """
    from sklearn.preprocessing import MinMaxScaler
    
    print("\n=== Calculating Normalized Feature Mean and Std for Each Cluster ===")
    print(f"Feature columns: {feature_columns}")
    print(f"Number of clusters: {len(cluster_ids)}")
    
    # Create DataFrame containing original features and cluster labels
    df_to_normalize = original_df[feature_columns].copy()
    df_to_normalize['cluster'] = cluster_labels
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize features to [0,1]
    normalized_features = scaler.fit_transform(df_to_normalize[feature_columns])
    df_normalized = pd.DataFrame(normalized_features, columns=feature_columns)
    df_normalized['cluster'] = cluster_labels
    
    # Compute mean and std of each feature for each cluster
    results = []
    for cluster_id in sorted(cluster_ids):
        cluster_data = df_normalized[df_normalized['cluster'] == cluster_id]
        
        # Compute mean and std for each feature in this cluster
        for feature in feature_columns:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            
            results.append({
                'Cluster': f'C{cluster_id}',
                'Feature': feature,
                'Normalized_Mean': mean_val,
                'Normalized_Std': std_val
            })
    
    # Create result DataFrame
    results_df = pd.DataFrame(results)
    
    # Reshape result for better readability
    pivot_mean = results_df.pivot(index='Cluster', columns='Feature', values='Normalized_Mean')
    pivot_std = results_df.pivot(index='Cluster', columns='Feature', values='Normalized_Std')
    
    print("\nNormalized feature means for each cluster (range [0,1]):")
    print(pivot_mean.round(4))
    
    print("\nNormalized feature standard deviations for each cluster:")
    print(pivot_std.round(4))
    
    # Save results to CSV files
    csv_mean_filename = "scripts/music_style_classification/results/normalized_feature_means_by_cluster_k=8.csv"
    csv_std_filename = "scripts/music_style_classification/results/normalized_feature_std_by_cluster_k=8.csv"
    pivot_mean.round(4).to_csv(csv_mean_filename)
    pivot_std.round(4).to_csv(csv_std_filename)
    print(f"\nSaved normalized feature means to: {csv_mean_filename}")
    print(f"Saved normalized feature standard deviations to: {csv_std_filename}")
    
    # Save combined results (including mean and std)
    combined_results = []
    for cluster in pivot_mean.index:
        for feature in pivot_mean.columns:
            combined_results.append({
                'Cluster': cluster,
                'Feature': feature,
                'Mean': pivot_mean.loc[cluster, feature],
                'Std': pivot_std.loc[cluster, feature],
                'Mean±Std': f"{pivot_mean.loc[cluster, feature]:.4f} ± {pivot_std.loc[cluster, feature]:.4f}"
            })
    
    combined_df = pd.DataFrame(combined_results)
    combined_filename = "scripts/music_style_classification/results/normalized_feature_stats_by_cluster_k=8.csv"
    combined_df.to_csv(combined_filename, index=False)
    print(f"Saved combined statistics to: {combined_filename}")
    
    return results_df

# Call function to compute statistics of normalized features
feature_columns = ['Danceability','Loudness','Speechiness', 'Acousticness','Instrumentalness','Valence']
cluster_ids = sorted(final_clustered_df['cluster_id'].unique())
normalized_stats = calculate_normalized_cluster_stats(
    df,  # Original DataFrame
    feature_columns,
    labels_kmpp,  # Use K-Means++ clustering results
    cluster_ids
)

print("\nClustering analysis completed!")
