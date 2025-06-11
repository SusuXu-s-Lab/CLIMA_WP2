# -*- coding: utf-8 -*-
"""
节点固定：以 intersected_home 为全集
----------------------------------
外部必须先定义：south, north, west, east 四个经纬度边界
"""
import glob, geohash
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


# ─────────── 工具函数 ───────────
def load_edges_by_month(pattern):
    out = {}
    for fp in sorted(glob.glob(pattern)):
        date = pd.to_datetime(Path(fp).stem.split('_')[-1])
        out[date] = pd.read_csv(fp)
    return out


def load_status(pattern):
    d = {}
    for fp in sorted(glob.glob(pattern)):
        date = pd.to_datetime(Path(fp).stem.split("home")[-1].lstrip("_"))
        d[date] = set(pd.read_csv(fp)["home_geohash_8"].unique())
    return d


# ─────────── 主函数 ───────────
def animate_status_over_time_dynamic(
    edge_pattern,
    migrated_pattern,
    vacant_pattern,
    intersected_home_path,              # ← 新增：交集节点 CSV
    out_gif="network_status.gif",
    base_size=20,
    size_scale=5,
    interval_ms=1000,
    bridging_keep_fraction=0.01,
    bonding_keep_fraction=1.0,
):
    # 1️⃣ intersected_home：节点全集 & 大小
    int_df = pd.read_csv(intersected_home_path,
                         usecols=["device_id", "home_geohash_8"])
    device_counts = (
        int_df.groupby("home_geohash_8")["device_id"]
              .nunique()
    )                                     # Series: home → #devices

    # geohash → (lat, lon) 并做 bbox 过滤
    centers = {
        h: geohash.decode(h)
        for h in device_counts.index
    }
    centers = {
        h: (lat, lon)
        for h, (lat, lon) in centers.items()
        if south <= lat <= north and west <= lon <= east
    }
    if not centers:
        raise ValueError("❌ intersected_home 中无节点落在 bounding box 内。")

    nodes = list(centers)                 # 固定顺序
    nodes_set = set(nodes)
    xs = [centers[h][1] for h in nodes]   # lon
    ys = [centers[h][0] for h in nodes]   # lat
    sizes_fixed = [
        base_size + size_scale * device_counts[h] for h in nodes
    ]

    # 2️⃣ 其他输入：边、状态
    edge_dict = load_edges_by_month(edge_pattern)
    if not edge_dict:
        raise ValueError("No edge files found.")
    months = sorted(edge_dict)

    mig_dict = load_status(migrated_pattern)
    vac_dict = load_status(vacant_pattern)

    # 3️⃣ 画布 & 静态节点
    fig, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter(
        xs, ys, s=sizes_fixed, c=["navy"] * len(nodes),
        edgecolors="black", alpha=0.8, zorder=3
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")

    legend_items = [
        Line2D([0], [0], color="crimson",   lw=2, label="Bridging links"),
        Line2D([0], [0], color="darkgreen", lw=2, label="Bonding links"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="navy",   markersize=8, label="Normal"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="red",    markersize=8, label="Migrated"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="orange", markersize=8, label="Vacant"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    edge_artists = []

    # 4️⃣ 帧更新
    def update(i):
        nonlocal edge_artists
        date = months[i]
        df = edge_dict[date].copy()

        # — bridging / bonding 抽样 —
        if bridging_keep_fraction < 1.0 or bonding_keep_fraction < 1.0:
            bri = df[df["type"] == 0]
            bon = df[df["type"] == 1]
            if bridging_keep_fraction < 1.0 and len(bri):
                bri = bri.sample(frac=bridging_keep_fraction, random_state=42)
            if bonding_keep_fraction < 1.0 and len(bon):
                bon = bon.sample(frac=bonding_keep_fraction, random_state=42)
            df = pd.concat([bri, bon], ignore_index=True)

        # — 只保留 intersected_home 里的边 —
        df = df[
            df["home_1"].isin(nodes_set) &
            df["home_2"].isin(nodes_set)
        ]

        # — 更新节点颜色 —
        mig_set = mig_dict.get(date, set())
        vac_set = vac_dict.get(date, set())
        scat.set_facecolors([
            "red"    if h in mig_set else
            "orange" if h in vac_set else
            "navy"
            for h in nodes
        ])

        # — 更新边 —
        for art in edge_artists:
            art.remove()
        edge_artists.clear()
        for _, r in df.iterrows():
            lat1, lon1 = centers[r["home_1"]]
            lat2, lon2 = centers[r["home_2"]]
            col = "crimson" if r["type"] == 0 else "darkgreen"
            ln, = ax.plot([lon1, lon2], [lat1, lat2],
                          color=col, alpha=0.6, lw=1, zorder=2)
            edge_artists.append(ln)

        ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)
        return edge_artists + [scat]

    # 5️⃣ 动画 & 保存
    anim = FuncAnimation(fig, update, frames=len(months),
                         interval=interval_ms, blit=True, repeat=True)
    anim.save(out_gif,
              writer="pillow" if out_gif.lower().endswith(".gif") else "ffmpeg",
              dpi=120)
    plt.close(fig)
    print(f"✅ Saved animation to {out_gif}")


# ───── 使用示例（请先设置 south / north / west / east） ─────
# south, north = 24.5, 31.0
# west,  east  = -88.0, -79.0
animate_status_over_time_dynamic(
    edge_pattern         = "home_tables_florida/Household_social_network_*.csv",
    migrated_pattern     = "home_tables_florida/Migrated_home*.csv",
    vacant_pattern       = "home_tables_florida/Vacant_home*.csv",
    intersected_home_path= "home_tables_florida/intersected_home.csv",
    out_gif              = "home_tables_florida/network_status.gif",
    base_size            = 30,
    size_scale           = 10,
    interval_ms          = 1000
)
