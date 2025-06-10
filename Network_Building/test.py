import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import geohash

# 定义边界框（需要根据你的数据调整）
south, north, west, east = 25.0, 31.0, -87.0, -80.0

def load_edges_by_month(edge_pattern: str):
    """Return {date -> edge-DF} where edge files are named like
       Household_social_network_2019-01-01.csv"""
    edge_dict = {}
    for fp in sorted(glob.glob(edge_pattern)):
        date_str = Path(fp).stem.split('_')[-1]          # "2019-01-01"
        date = pd.to_datetime(date_str)
        edge_dict[date] = pd.read_csv(fp)
    return edge_dict

def build_node_info(df_edges):
    """
    Return (home_center_dict, home_device_count) for a single month.

    - Skips NaN values in home_2.
    - Merges the *_number columns into one device-count dictionary.
    """
    # ---------------- 1. Collect node IDs ----------------
    nodes = pd.unique(
        pd.concat([df_edges['home_1'], df_edges['home_2']])
          .dropna()                             # drop NaN rows
    )

    # ---------------- 2. Geohash → (lat, lon) ----------------
    centers = {
        h: geohash.decode(h) for h in nodes if isinstance(h, str) and h.strip()
    }

    # ---------------- 3. Device counts per node ----------------
    s1 = (df_edges[['home_1', 'home_1_number']]
          .dropna(subset=['home_1'])            # keep rows with a home_1 id
          .drop_duplicates()
          .set_index('home_1')['home_1_number'])

    s2 = (df_edges[['home_2', 'home_2_number']]
          .dropna(subset=['home_2'])            # drop NaN home_2 rows
          .drop_duplicates()
          .set_index('home_2')['home_2_number'])

    counts = pd.concat([s1, s2]).to_dict()

    return centers, counts

def animate_status_over_time_dynamic(
    edge_pattern: str,
    migrated_pattern: str,
    vacant_pattern: str,
    out_gif: str = "network_status.gif",
    base_size: float = 20.0,
    size_scale: float = 5.0,
    interval_ms: int = 1000,
    bridging_keep_fraction: float = 0.01,   # <── NEW
    bonding_keep_fraction:  float = 1.0    # <── NEW
):
    # ───────── 1. Load per-month data ─────────
    edge_dict = load_edges_by_month(edge_pattern)

    def load_status(pattern):
        d = {}
        for fp in sorted(glob.glob(pattern)):
            date_str = Path(fp).stem.split("home")[-1].lstrip("_")
            date = pd.to_datetime(date_str)
            homes = pd.read_csv(fp)["home_geohash_8"].unique()
            d[date] = set(homes)
        return d
    mig_dict = load_status(migrated_pattern)
    vac_dict = load_status(vacant_pattern)

    months = sorted(edge_dict)              # timeline
    if not months:
        raise ValueError("No edge files found.")

    # ───────── 2. 预先收集所有节点 ─────────
    all_centers = {}
    all_counts = {}
    
    for date, df_e in edge_dict.items():
        center, counts = build_node_info(df_e)
        # 应用边界框过滤
        center = {
            h: (lat, lon) for h, (lat, lon) in center.items()
            if south <= lat <= north and west <= lon <= east
        }
        all_centers.update(center)
        all_counts.update(counts)

    # ───────── 3. Build figure 并预先绘制所有节点 ─────────
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 预先绘制所有节点（初始状态）
    xs = [lon for lat, lon in all_centers.values()]
    ys = [lat for lat, lon in all_centers.values()]
    sizes = [base_size + size_scale * all_counts.get(hid, 1) for hid in all_centers.keys()]
    colors = ["navy"] * len(all_centers)  # 初始颜色
    
    # 创建scatter plot对象，后续只更新其属性
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="black", alpha=0.8)
    
    # 创建空的线条列表，用于存储边
    lines = []

    # Custom legend (fixed)
    legend_items = [
        Line2D([0], [0], color="crimson",  lw=2, label="Bridging links"),
        Line2D([0], [0], color="darkgreen",lw=2, label="Bonding links"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="navy",   markersize=8, label="Normal"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="red",    markersize=8, label="Migrated"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="orange", markersize=8, label="Vacant")
    ]
    ax.legend(handles=legend_items, loc="upper right")

    def update(frame_idx):
        date   = months[frame_idx]
        df_e   = edge_dict[date]
        
        # 采样处理
        if bridging_keep_fraction < 1.0 or bonding_keep_fraction < 1.0:
            bri = df_e[df_e['type'] == 0]      # bridging
            bon = df_e[df_e['type'] == 1]      # bonding
            if bridging_keep_fraction < 1.0 and len(bri):
                bri = bri.sample(
                    frac=bridging_keep_fraction, random_state=42)
            if bonding_keep_fraction < 1.0 and len(bon):
                bon = bon.sample(
                    frac=bonding_keep_fraction,  random_state=42)
            df_e = pd.concat([bri, bon], ignore_index=True)
            
        # 过滤边数据，只保留在all_centers中的节点
        df_e = df_e[
            df_e['home_1'].isin(all_centers) &
            (df_e['home_2'].isna() | df_e['home_2'].isin(all_centers))
        ].reset_index(drop=True)
        
        mig_set = mig_dict.get(date, set())
        vac_set = vac_dict.get(date, set())

        # 清除之前的边
        for line in lines:
            line.remove()
        lines.clear()

        # 更新标题
        ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)

        # -------- 绘制新的边 --------
        for _, row in df_e.iterrows():
            if pd.isna(row['home_2']):          # <── NEW guard
                continue                        # skip "edge" with no target
            if row['home_1'] in all_centers and row['home_2'] in all_centers:
                lat1, lon1 = all_centers[row['home_1']]
                lat2, lon2 = all_centers[row['home_2']]
                col = "crimson" if row['type'] == 0 else "darkgreen"
                line, = ax.plot([lon1, lon2], [lat1, lat2], color=col, alpha=0.6, lw=1)
                lines.append(line)

        # -------- 更新节点颜色 --------
        new_colors = []
        for hid in all_centers.keys():
            if hid in mig_set:       new_colors.append("red")
            elif hid in vac_set:     new_colors.append("orange")
            else:                    new_colors.append("navy")
        
        scatter.set_color(new_colors)

        return [scatter] + lines

    anim = FuncAnimation(
        fig, update, frames=len(months),
        interval=interval_ms, repeat=True, blit=False
    )

    # ───────── 4. Save ─────────
    if out_gif.lower().endswith(".gif"):
        anim.save(out_gif, writer="pillow", dpi=120)
    else:
        anim.save(out_gif, writer="ffmpeg", dpi=120)

    plt.close(fig)
    print(f"✅ Saved animation to {out_gif}")

animate_status_over_time_dynamic(
    edge_pattern="florida_results/Household_social_network_*.csv",
    migrated_pattern="florida_results/Migrated_home*.csv",
    vacant_pattern="florida_results/Vacant_home*.csv",
    out_gif="florida_results/network_status.gif",
    base_size=30,
    size_scale=10,
    interval_ms=1000
)
