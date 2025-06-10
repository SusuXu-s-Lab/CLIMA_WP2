import glob, geohash
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


# ───── 工具函数 ─────
def load_edges_by_month(pattern):
    out = {}
    for fp in sorted(glob.glob(pattern)):
        date = pd.to_datetime(Path(fp).stem.split('_')[-1])
        out[date] = pd.read_csv(fp)
    return out


def build_node_info(df):
    nodes = pd.concat([df['home_1'], df['home_2']]).dropna().unique()
    centers = {h: geohash.decode(h) for h in nodes if isinstance(h, str) and h.strip()}
    s1 = (df[['home_1', 'home_1_number']]
          .dropna(subset=['home_1']).drop_duplicates()
          .set_index('home_1')['home_1_number'])
    s2 = (df[['home_2', 'home_2_number']]
          .dropna(subset=['home_2']).drop_duplicates()
          .set_index('home_2')['home_2_number'])
    counts = pd.concat([s1, s2]).to_dict()
    return centers, counts


def load_status(pattern):
    d = {}
    for fp in sorted(glob.glob(pattern)):
        date = pd.to_datetime(Path(fp).stem.split("home")[-1].lstrip("_"))
        d[date] = set(pd.read_csv(fp)["home_geohash_8"].unique())
    return d


# ───── 主函数 ─────
def animate_status_over_time_dynamic(
    edge_pattern, migrated_pattern, vacant_pattern,
    out_gif="network_status.gif",
    base_size=20, size_scale=5, interval_ms=1000,
    bridging_keep_fraction=0.01, bonding_keep_fraction=1.0
):
    edge_dict = load_edges_by_month(edge_pattern)
    if not edge_dict:
        raise ValueError("No edge files found.")
    months = sorted(edge_dict)

    # —— 全局节点坐标 + 最大设备数（固定大小用） ——
    union_centers, max_counts = {}, {}
    for df in edge_dict.values():
        ctr, cnt = build_node_info(df)
        for h, (lat, lon) in ctr.items():
            if south <= lat <= north and west <= lon <= east:
                union_centers[h] = (lat, lon)
                max_counts[h] = max(max_counts.get(h, 0), cnt.get(h, 0))
    if not union_centers:
        raise ValueError("No nodes inside bounding box.")

    nodes = list(union_centers)
    xs = [union_centers[h][1] for h in nodes]   # lon
    ys = [union_centers[h][0] for h in nodes]   # lat
    sizes_fixed = [
        base_size + size_scale * max_counts.get(h, 1) for h in nodes
    ]

    mig_dict = load_status(migrated_pattern)
    vac_dict = load_status(vacant_pattern)

    # —— 画布 & 静态节点 ——
    fig, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter(
        xs, ys, s=sizes_fixed, c=["navy"]*len(nodes),
        edgecolors="black", alpha=0.8, zorder=3
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect('equal', adjustable='box')

    legend_items = [
        Line2D([0], [0], color="crimson", lw=2, label="Bridging links"),
        Line2D([0], [0], color="darkgreen", lw=2, label="Bonding links"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="navy", markersize=8, label="Normal"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="red",  markersize=8, label="Migrated"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="orange", markersize=8, label="Vacant"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    edge_artists = []

    # —— 帧更新 —— 
    def update(i):
        nonlocal edge_artists
        date = months[i]
        df = edge_dict[date].copy()

        # 抽样保留
        if bridging_keep_fraction < 1.0 or bonding_keep_fraction < 1.0:
            bri = df[df['type'] == 0]
            bon = df[df['type'] == 1]
            if bridging_keep_fraction < 1.0 and len(bri):
                bri = bri.sample(frac=bridging_keep_fraction, random_state=42)
            if bonding_keep_fraction < 1.0 and len(bon):
                bon = bon.sample(frac=bonding_keep_fraction, random_state=42)
            df = pd.concat([bri, bon], ignore_index=True)

        # 过滤 bbox 外结点
        df = df[
            df['home_1'].isin(union_centers) &
            (df['home_2'].isna() | df['home_2'].isin(union_centers))
        ]

        # —— 仅更新颜色 —— 
        mig_set = mig_dict.get(date, set())
        vac_set = vac_dict.get(date, set())
        scat.set_facecolors([
            "red" if h in mig_set else
            "orange" if h in vac_set else
            "navy"
            for h in nodes
        ])

        # —— 更新边 —— 
        for art in edge_artists:
            art.remove()
        edge_artists.clear()
        for _, r in df.iterrows():
            if pd.isna(r['home_2']):
                continue
            lat1, lon1 = union_centers[r['home_1']]
            lat2, lon2 = union_centers[r['home_2']]
            col = "crimson" if r['type'] == 0 else "darkgreen"
            ln, = ax.plot([lon1, lon2], [lat1, lat2],
                          color=col, alpha=0.6, lw=1, zorder=2)
            edge_artists.append(ln)

        ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)
        return edge_artists + [scat]

    anim = FuncAnimation(fig, update, frames=len(months),
                         interval=interval_ms, blit=True, repeat=True)

    anim.save(out_gif,
              writer="pillow" if out_gif.lower().endswith(".gif") else "ffmpeg",
              dpi=120)
    plt.close(fig)
    print(f"✅ Saved animation to {out_gif}")


# —— 使用示例（确保 south/north/west/east 先定义） ——
# south, north = 24.5, 31.0
# west,  east  = -88.0, -79.0
animate_status_over_time_dynamic(
    edge_pattern     = "florida_results/Household_social_network_*.csv",
    migrated_pattern = "florida_results/Migrated_home*.csv",
    vacant_pattern   = "florida_results/Vacant_home*.csv",
    out_gif          = "florida_results/network_status.gif",
    base_size        = 30,
    size_scale       = 10,
    interval_ms      = 1000
)
