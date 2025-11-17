import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


def save_image(fig, filename, width=None, height=None, scale=2):
    import tempfile, time, os, io
    from pathlib import Path
    import plotly.io as pio

    html = pio.to_html(fig, include_plotlyjs=True, full_html=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        html_path = f.name

 # 2) Start headless browser
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    try:
        from selenium.webdriver.chrome.options import Options
        options = Options()
        browser = "chrome"
    except Exception:
        raise

  # Set window size 
    W = int((width or 1400) * max(1, scale))
    H = int((height or 900) * max(1, scale))
    options.add_argument("--headless=new")
    options.add_argument(f"--window-size={W},{H}")

    driver = None
    try:
        try:
            driver = webdriver.Chrome(options=options)
        except Exception:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

        driver.get(Path(html_path).resolve().as_uri())
        time.sleep(1.2) 

        el = driver.find_element(By.CSS_SELECTOR, "div.js-plotly-plot")
        png_bytes = el.screenshot_as_png  

        out_path = Path(OUT_DIR) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ext = out_path.suffix.lower()
        if ext in [".png"]:
            with open(out_path, "wb") as out:
                out.write(png_bytes)
        elif ext in [".jpg", ".jpeg"]:
            try:
                from PIL import Image
            except Exception:
                raise RuntimeError("Exporting JPG requires pillow: pip install pillow")
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            img.save(out_path, format="JPEG", quality=95)
        else:
            raise ValueError("filename must end with .png or .jpg")

    finally:
        if driver is not None:
            driver.quit()
        try:
            os.remove(html_path)
        except Exception:
            pass

# data
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "data_preparation"

CSV_PATH = DATA_DIR / "data_clean_new.csv"
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV_PATH)


def pick_first(cols, candidates):
    # exact match
    for c in candidates:
        for col in cols:
            if col.lower() == c.lower():
                return col
    # contains fallback
    lc = {col.lower(): col for col in cols}
    for c in candidates:
        for k, v in lc.items():
            if c in k:
                return v
    return None

def first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# columns 
cols = list(df.columns)
lc = {c.lower(): c for c in cols}

title_col  = pick_first(cols, ["title","track","track_name","song","name"])
artist_col = pick_first(cols, ["artist","artists","primary_artist","artist_name"])
genre_col  = pick_first(cols, ["genre","genres","primary_genre"])
year_col   = pick_first(cols, ["year"])
date_col   = pick_first(cols, ["date","release_date","chart_date"])

popularity_col = pick_first(cols, ["popularity","pop_points_total","pop_points_artist"])
rank_col       = pick_first(cols, ["rank","chart_rank","position"])
explicit_col   = pick_first(cols, ["explicit","is_explicit"])

feature_keys = ["danceability","energy","loudness","speechiness","acousticness",
                "instrumentalness","liveness","valence","tempo"]
present_features = [lc[k] for k in feature_keys if k in lc]

if (year_col is None) and (date_col is not None) and (date_col in df.columns):
    tmp_dt = pd.to_datetime(df[date_col], errors="coerce")
    if tmp_dt.notna().any():
        df["_year"] = tmp_dt.dt.year
        year_col = "_year"


# correlation_heatmap
numeric_df = df.select_dtypes(include=np.number)
to_drop = [c for c in numeric_df.columns if c.lower() in {"year", "_year", "month"}]
numeric_df = numeric_df.drop(columns=to_drop, errors="ignore")

if numeric_df.shape[1] >= 2:
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    numeric_df = numeric_df.loc[:, numeric_df.var(axis=0, numeric_only=True) > 0]
    corr = numeric_df.corr(numeric_only=True)

    k = corr.shape[0]
    fig_side = int(min(1400, max(800, k * 34)))

    fig = px.imshow(
        corr,
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        text_auto=".2f",
        aspect='auto',
        labels=dict(color="ρ"),
        title="<b>Pairwise Correlation of Numerical Columns</b>"
    )
    fig.update_layout(
        template="plotly_white",
        width=fig_side, height=fig_side,
        title_x=0.5,
        font=dict(size=13),
        margin=dict(l=90, r=90, t=100, b=80),
        coloraxis_colorbar=dict(title="ρ", tickvals=[-1,-0.5,0,0.5,1], ticks="outside", len=0.8),
    )
    fig.update_xaxes(tickangle=45, showgrid=False, zeroline=False, mirror=True, showline=True, linewidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, mirror=True, showline=True, linewidth=1)
    fig.update_traces(hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>")

    save_image(fig, "correlation_heatmap.png", width=fig_side, height=fig_side, scale=2)

print(f"✅ Images saved to: {OUT_DIR}")