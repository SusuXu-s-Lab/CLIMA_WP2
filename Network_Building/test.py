import glob, geohash
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ---------------- util ----------------
def load_edges_by_month(edge_pattern: str):
    out = {}
    for fp in sorted(glob.glob(edge_pattern)):
        date = pd.to_datetime(Path(fp).stem.split('_')[-1])
        out[date] = pd.read_csv(fp)
    return out

def merge_node_info(edge_dict):
    """一次性合并所有月份中的 node → (lat,lon) 与设备数"""
    centers, counts = {}, {}
    for df in edge_dict.values():
        # 结点 id
        nodes = pd.unique(pd.concat([df['home_1'], df['home_2']]).dropna())
        # 坐标
        centers.update({h: geohash.decode(h) for h in nodes
                        if isinstance(h, str) and h.strip()})
        # 设备数
        s1 = (df[['home_1', 'home_1_number']]
              .dropna(subset=['home_1']).drop_duplicates()
              .set_index('home_1')['home_1_number'])
        s2 = (df[['home_2', 'home_2_number']]
              .dropna(subset=['home_2']).drop_duplicates()
              .set_index('home_2')['home_2_number'])
        counts.update(pd.concat([s1, s2]).to_dict())
    return centers, counts

# ---------------- main ----------------
def animate_status_over_time_static_nodes(
    edge_pattern, migrated_pattern, vacant_pattern,
    out_gif="network_status.gif", base_size=20, size_scale=5,
    interval_ms=1000, bridging_keep_fraction=0.01, bonding_keep_fraction=1.0
):
    # ① 读所有月份
    edge_dict = load_edges_by_month(edge_pattern)
    if not edge_dict:
        raise ValueError("No edge files found")

    # ② 读迁移/空置
    def load_status(pattern):
        d = {}
        for fp in sorted(glob.glob(pattern)):
            date = pd.to_datetime(Path(fp).stem.split("home")[-1].lstrip("_"))
            d[date] = set(pd.read_csv(fp)["home_geohash_8"].unique())
        return d
    mig_dict, vac_dict = map(load_status, (migrated_pattern, vacant_pattern))

    # ③ 全量节点信息 + 过滤到 bbox（这里示例用全部坐标；如有 south/north 等自己加）
    centers, counts = merge_node_info(edge_dict)
    xs, ys = zip(*[(lon, lat) for lat, lon in centers.values()])

    # ④ 预画静态节点
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = [base_size + size_scale * counts.get(h, 1) for h in centers]
    node_colors = ["navy"] * len(centers)  # placeholder，会在 update 里改
    scat = ax.scatter(xs, ys, s=sizes, c=node_colors,
                      edgecolors="black", alpha=0.8, zorder=3)

    legend_items = [
        Line2D([0], [0], color="crimson", lw=2, label="Bridging links"),
        Line2D([0], [0], color="darkgreen", lw=2, label="Bonding links"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="navy", markersize=8, label="Normal"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="red", markersize=8, label="Migrated"),
        Line2D([0], [0], marker="o", color="black",
               markerfacecolor="orange", markersize=8, label="Vacant"),
    ]
    ax.legend(handles=legend_items, loc="upper right")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # 用于存放本帧画出的边，以便下一帧删除
    edge_artists = []

    months = sorted(edge_dict)
    def update(i):
        nonlocal edge_artists
        date  = months[i]
        df_e  = edge_dict[date]

        # 可选抽样保留
        if bridging_keep_fraction < 1.0 or bonding_keep_fraction < 1.0:
            bri = df_e[df_e['type'] == 0]
            bon = df_e[df_e['type'] == 1]
            if bridging_keep_fraction < 1.0 and len(bri):
                bri = bri.sample(frac=bridging_keep_fraction, random_state=42)
            if bonding_keep_fraction < 1.0 and len(bon):
                bon = bon.sample(frac=bonding_keep_fraction, random_state=42)
            df_e = pd.concat([bri, bon], ignore_index=True)

        # ----- 更新节点颜色 -----
        mig_set = mig_dict.get(date, set())
        vac_set = vac_dict.get(date, set())
        new_cols = []
        for hid in centers:
            if hid in mig_set:   new_cols.append("red")
            elif hid in vac_set: new_cols.append("orange")
            else:                new_cols.append("navy")
        scat.set_facecolors(new_cols)

        # ----- 更新边 -----
        for art in edge_artists:  # 清掉上一帧的线
            art.remove()
        edge_artists = []
        for _, row in df_e.iterrows():
            if pd.isna(row['home_2']):  # 跳过单端
                continue
            lat1, lon1 = centers[row['home_1']]
            lat2, lon2 = centers[row['home_2']]
            col = "crimson" if row['type'] == 0 else "darkgreen"
            ln, = ax.plot([lon1, lon2], [lat1, lat2],
                          color=col, alpha=0.6, lw=1, zorder=2)
            edge_artists.append(ln)

        ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)
        return edge_artists + [scat]

    anim = FuncAnimation(fig, update, frames=len(months),
                         interval=interval_ms, blit=True, repeat=True)

    anim.save(out_gif, writer="pillow" if out_gif.endswith(".gif") else "ffmpeg", dpi=120)
    plt.close(fig)
    print(f"✅ Saved animation to {out_gif}")

# ---------- 调用 ----------
animate_status_over_time_static_nodes(
    edge_pattern="florida_results/Household_social_network_*.csv",
    migrated_pattern="florida_results/Migrated_home*.csv",
    vacant_pattern="florida_results/Vacant_home*.csv",
    out_gif="florida_results/network_status.gif",
    base_size=30, size_scale=10, interval_ms=1000
)
