import numpy as np
import pandas as pd

raw = {
    "C0":{"name":"Balanced Mix","Danceability":(-1.2928,0.6959),"Loudness":(-0.0421,0.7118),
          "Speechiness":(-0.4227,0.5054),"Acousticness":(-0.4627,0.5544),
          "Instrumentalness":(-0.0839,0.3279),"Valence":(-0.7072,0.7231)},
    "C1":{"name":"Energetic Pop","Danceability":(0.4985,0.5651),"Loudness":(-0.3059,0.5987),
          "Speechiness":(-0.1375,0.5721),"Acousticness":(-0.4019,0.5638),
          "Instrumentalness":(-0.0995,0.2733),"Valence":(-0.6953,0.5828)},
    "C2":{"name":"Instrumental Beats","Danceability":(-1.1806,1.6036),"Loudness":(-1.2787,1.8401),
          "Speechiness":(-0.3016,1.1593),"Acousticness":(0.6586,1.5307),
          "Instrumentalness":(12.2716,1.6381),"Valence":(-0.8075,0.9845)},
    "C3":{"name":"Hip-hop","Danceability":(0.2213,0.9952),"Loudness":(-0.2697,0.9093),
          "Speechiness":(2.2474,0.9485),"Acousticness":(-0.1321,0.8475),
          "Instrumentalness":(-0.1352,0.1468),"Valence":(-0.0460,0.8616)},
    "C4":{"name":"Acoustic","Danceability":(-1.1505,0.9970),"Loudness":(-1.4797,1.1756),
          "Speechiness":(-0.4555,0.5870),"Acousticness":(2.0840,0.6839),
          "Instrumentalness":(-0.0241,0.4554),"Valence":(-0.6465,0.8880)},
    "C5":{"name":"Big-Room Dance","Danceability":(0.0613,0.8218),"Loudness":(1.7451,0.3623),
          "Speechiness":(-0.0805,0.7501),"Acousticness":(-0.0263,0.9668),
          "Instrumentalness":(-0.1027,0.2779),"Valence":(0.0525,0.9200)},
    "C6":{"name":"Mellow Blend","Danceability":(-0.4565,1.3000),"Loudness":(-0.6718,1.3065),
          "Speechiness":(-0.2503,0.7450),"Acousticness":(0.2635,1.3025),
          "Instrumentalness":(5.7386,1.7120),"Valence":(-0.7807,0.9902)},
    "C7":{"name":"Latin Pop","Danceability":(0.4056,0.6773),"Loudness":(0.0845,0.6248),
          "Speechiness":(-0.2691,0.5424),"Acousticness":(-0.1746,0.7316),
          "Instrumentalness":(-0.1188,0.2139),"Valence":(1.0052,0.5123)}
}

features = ["Danceability","Loudness","Speechiness","Acousticness","Instrumentalness","Valence"]

def scale_means_sds(raw, features, use_ci95=False, n=None, z=1.96, clip=True):
    clusters = list(raw.keys())
    F = len(features)

    def halfwidth(k, feat):
        mean, sd = raw[k][feat]
        if use_ci95:
            if n is None or k not in n:
                raise ValueError("Provide sample sizes n per cluster for CI.")
            return z * sd / np.sqrt(n[k])
        return sd

    # per-feature global min/max from (mean ± band)
    mins = np.zeros(F); maxs = np.zeros(F)
    for j, feat in enumerate(features):
        lows = [raw[k][feat][0] - halfwidth(k, feat) for k in clusters]
        ups  = [raw[k][feat][0] + halfwidth(k, feat) for k in clusters]
        mins[j], maxs[j] = np.min(lows), np.max(ups)
        if np.isclose(maxs[j]-mins[j], 0):  # safety
            maxs[j] = mins[j] + 1.0
    scale = maxs - mins

    # scaled mean and sd (or CI half-width if use_ci95=True)
    mean_scaled = pd.DataFrame(index=clusters, columns=features, dtype=float)
    sd_scaled   = pd.DataFrame(index=clusters, columns=features, dtype=float)

    for k in clusters:
        for j, feat in enumerate(features):
            mean, sd = raw[k][feat]
            band = halfwidth(k, feat)
            ms = (mean - mins[j]) / scale[j]
            sds = band / scale[j] if use_ci95 else sd / scale[j]
            if clip:
                ms = np.clip(ms, 0, 1)
            mean_scaled.at[k, feat] = ms
            sd_scaled.at[k, feat]   = sds

    mins_s = pd.Series(mins, index=features, name="global_min")
    maxs_s = pd.Series(maxs, index=features, name="global_max")
    return mean_scaled, sd_scaled, mins_s, maxs_s


mean_scaled, sd_scaled, global_min, global_max = scale_means_sds(raw, features)
print("Scaled means:\n", mean_scaled.round(4))
print("\nScaled sds (per-feature scaled):\n", sd_scaled.round(4))
mean_scaled.to_excel(r'Kmeans++8_Feature_Mean_Std\feature_scaled_mean.xlsx')
sd_scaled.to_excel(r'Kmeans++8_Feature_Mean_Std\feature_scaled_std.xlsx')
