
# Donut pie chart for C1–C8 with latest data; try Arial, fallback to DejaVu Sans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# Latest cluster sizes (including C8)
#cluster_sizes = {1: 1231, 2: 60, 3: 190, 4: 1913, 5: 1151, 6: 1790, 7: 688, 8: 1670}
cluster_sizes = {1: 1465, 2: 1636, 3: 69, 4: 1913, 5: 969, 6: 870, 7: 109, 8: 2130}

# Keep C1–C8 in ascending order
cluster_sizes_filtered = {k: cluster_sizes[k] for k in sorted(cluster_sizes.keys()) if 1 <= k <= 8}
labels = [f"C{k}" for k in cluster_sizes_filtered.keys()]
sizes = np.array(list(cluster_sizes_filtered.values()), dtype=float)

total = sizes.sum()
percentages = sizes / total * 100

# Legend labels with counts and percentages
legend_labels = [f"{lab}: {int(cnt)} ({pct:.1f}%)" for lab, cnt, pct in zip(labels, sizes, percentages)]

# Decide on font: prefer Arial, else DejaVu Sans
arial_available = any("Arial" in f.name for f in fm.fontManager.ttflist)
preferred_font = "Arial" if arial_available else "DejaVu Sans"

plt.rcParams.update({
    "font.family": preferred_font,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "font.size": 18,
    "figure.dpi": 100
})

# Professional palette (tab10)
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(labels))]

fig, ax = plt.subplots(figsize=(7.5, 6.2))

wedges, texts = ax.pie(
    sizes,
    labels=None,
    startangle=90,
    counterclock=False,
    colors=colors,
    labeldistance=1.05,
    wedgeprops={"edgecolor": "#1a1a1a", "linewidth": 0.8},
    normalize=True
)

centre_circle = plt.Circle((0, 0), 0.60, fc="white", edgecolor="#1a1a1a", linewidth=1.2)
ax.add_artist(centre_circle)

ax.set_aspect("equal")

ax.legend(
    wedges,
    legend_labels,
    title="C1–C8 Breakdown",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

ax.set_title("Hierarchical Cluster Distribution (C1–C8)")

ax.text(0, -1.25, f"Total (C1–C8): {int(total)}", ha="center", va="center", fontsize=10, color="#4d4d4d")



plt.show()
