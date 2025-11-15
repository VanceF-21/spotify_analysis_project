# Retry: Radar charts (2x4) with min–max scaling to [0,1]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

# Feature order (top starts with Danceability)
features = ["Danceability", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Valence"]

raw = {
    "C0": {"name":"Balanced Mix","Danceability":(-1.2928,0.6959),"Loudness":(-0.0421,0.7118),
           "Speechiness":(-0.4227,0.5054),"Acousticness":(-0.4627,0.5544),
           "Instrumentalness":(-0.0839,0.3279),"Valence":(-0.7072,0.7231)},
    "C1": {"name":"Energetic Pop","Danceability":(0.4985,0.5651),"Loudness":(-0.3059,0.5987),
           "Speechiness":(-0.1375,0.5721),"Acousticness":(-0.4019,0.5638),
           "Instrumentalness":(-0.0995,0.2733),"Valence":(-0.6953,0.5828)},
    "C2": {"name":"Instrumental Beats","Danceability":(-1.1806,1.6036),"Loudness":(-1.2787,1.8401),
           "Speechiness":(-0.3016,1.1593),"Acousticness":(0.6586,1.5307),
           "Instrumentalness":(12.2716,1.6381),"Valence":(-0.8075,0.9845)},
    "C3": {"name":"Hip-hop","Danceability":(0.2213,0.9952),"Loudness":(-0.2697,0.9093),
           "Speechiness":(2.2474,0.9485),"Acousticness":(-0.1321,0.8475),
           "Instrumentalness":(-0.1352,0.1468),"Valence":(-0.046,0.8616)},
    "C4": {"name":"Acoustic","Danceability":(-1.1505,0.997),"Loudness":(-1.4797,1.1756),
           "Speechiness":(-0.4555,0.587),"Acousticness":(2.084,0.6839),
           "Instrumentalness":(-0.0241,0.4554),"Valence":(-0.6465,0.888)},
    "C5": {"name":"Big-Room Dance","Danceability":(0.0613,0.8218),"Loudness":(1.7451,0.3623),
           "Speechiness":(-0.0805,0.7501),"Acousticness":(-0.0263,0.9668),
           "Instrumentalness":(-0.1027,0.2779),"Valence":(0.0525,0.92)},
    "C6": {"name":"Mellow Blend","Danceability":(-0.4565,1.3),"Loudness":(-0.6718,1.3065),
           "Speechiness":(-0.2503,0.745),"Acousticness":(0.2635,1.3025),
           "Instrumentalness":(5.7386,1.712),"Valence":(-0.7807,0.9902)},
    "C7": {"name":"Latin Pop","Danceability":(0.4056,0.6773),"Loudness":(0.0845,0.6248),
           "Speechiness":(-0.2691,0.5424),"Acousticness":(-0.1746,0.7316),
           "Instrumentalness":(-0.1188,0.2139),"Valence":(1.0052,0.5123)},
}

# Per-feature min and max from (mean ± sd) across clusters
F = len(features)
cluster_keys = list(raw.keys())
mins = np.zeros(F); maxs = np.zeros(F)
for j, feat in enumerate(features):
    lowers = []; uppers = []
    for k in cluster_keys:
        m, s = raw[k][feat]
        lowers.append(m - s); uppers.append(m + s)
    mins[j] = np.min(lowers); maxs[j] = np.max(uppers)
    if np.isclose(maxs[j]-mins[j], 0): maxs[j] = mins[j] + 1.0

def scale_val(x, j): return (x - mins[j]) / (maxs[j] - mins[j])

scaled_means = {k: np.zeros(F) for k in cluster_keys}
scaled_lowers = {k: np.zeros(F) for k in cluster_keys}
scaled_uppers = {k: np.zeros(F) for k in cluster_keys}

for k in cluster_keys:
    for j, feat in enumerate(features):
        m, s = raw[k][feat]; lo, up = m - s, m + s
        scaled_means[k][j]  = np.clip(scale_val(m, j), 0, 1)
        scaled_lowers[k][j] = np.clip(scale_val(lo, j), 0, 1)
        scaled_uppers[k][j] = np.clip(scale_val(up, j), 0, 1)
print(scaled_means)
# Plot settings
arial_available = any("Arial" in f.name for f in fm.fontManager.ttflist)
preferred_font = "Arial" if arial_available else "DejaVu Sans"
plt.rcParams.update({
    "font.family": preferred_font,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "font.size": 13,
    "figure.dpi": 200
})

K = len(features)
thetas = np.linspace(0, 2*np.pi, K, endpoint=False)
thetas = np.concatenate([thetas, [thetas[0]]])

def plot_radar(ax, mean_vals, lower_vals, upper_vals, color, title):
    mean = np.array(mean_vals); lower = np.array(lower_vals); upper = np.array(upper_vals)
    mean_c = np.concatenate([mean, [mean[0]]])
    lower_c = np.concatenate([lower, [lower[0]]])
    upper_c = np.concatenate([upper, [upper[0]]])

    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_ylim(0,1)
    ax.set_xticks(thetas[:-1]); ax.set_xticklabels(features, color="black"); ax.tick_params(axis='x', pad=3)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0]); ax.set_yticklabels([])
    ax.grid(color="black", linestyle="--", linewidth=0.6)
    for spine in ax.spines.values(): spine.set_color("black"); spine.set_linewidth(0.9)

    theta_band = np.concatenate([thetas, thetas[::-1]])
    r_band = np.concatenate([upper_c, lower_c[::-1]])
    ax.fill(theta_band, r_band, color="#d0d0d0", alpha=0.5, linewidth=0)

    ax.plot(thetas, mean_c, color=color, linewidth=2.0)
    ax.fill(thetas, mean_c, color=color, alpha=0.28, edgecolor=color, linewidth=1.0)
    ax.set_title(title, pad=12, color="black")

# 2x4 grid
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(2, 4, subplot_kw=dict(polar=True), figsize=(13, 7.2))
axes = axes.ravel()
order = ["C0","C1","C2","C3","C4","C5","C6","C7"]
for i, key in enumerate(order):
    plot_radar(axes[i], scaled_means[key], scaled_lowers[key], scaled_uppers[key], cmap(i%10), f"{key}: {raw[key]['name']}")

# Shared legend
handles = [Patch(facecolor="#d0d0d0", edgecolor="none", label="± SD (band)"),
           Patch(facecolor=cmap(0), edgecolor=cmap(0), alpha=0.28, label="Mean (fill)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0, 0, 1, 0.98])


# Save
fig.savefig("cluster_radar_2x4.png", dpi=300, bbox_inches="tight")
# fig.savefig("cluster_radar_2x4.pdf", bbox_inches="tight")

plt.show()
