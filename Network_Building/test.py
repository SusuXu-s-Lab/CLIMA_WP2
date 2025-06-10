
import glob, geohash
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ────────────────────────────── 工具函数 ──────────────────────────────
def load_edges_by_month(edge_pattern: str):
    """读取每月网络边 CSV，返回 {date → DataFrame}。"""
    edge_dict = {}
    for fp in sorted(glob.glob(edge_pattern)):
        date_str = Path(fp).stem.split('_')[-1]     # e.g. 2019-01-01
        edge_dict[pd.to_datetime(date_str)] = pd.read_csv(fp)
    return edge_dict


def build_node_info(df_edges: pd.DataFrame):
    """给定某月边表，返回 (center_dict, count_dict)。"""
    # 1. 取所有结点 ID
    nodes = pd.concat([df_edges['home_1'], df_edges['home_2']]).dropna().unique()

    # 2. geohash → (lat, lon)
    centers = {h: geohash.decode(h) for h in nodes if isinstance(h, str) and h.strip()}

    # 3. 设备数量
    s1 = (df_edges[['home_1', 'home_1_number']]
          .dropna(subset=['home_1']).drop_duplicates()
          .set_index('home_1')['home_1_number'])
    s2 = (df_edges[['home_2', 'home_2_number']]
          .dropna(subset=['home_2']).drop_duplicates()
          .set_index('home_2')['home_2_number'])
    counts = pd.concat([s1, s2]).to_dict()
    return centers, counts


def load_status_dict(pattern: str):
    """读取 Migrated_home* / Vacant_home*，返回 {date → set(home_geohash_8)}。"""
    dct = {}
    for fp in sorted(glob.glob(pattern)):
        date = pd.to_datetime(Path(fp).stem.split("home")[-1].lstrip("_"))
        dct[date] = set(pd.read_csv(fp)["home_geohash_8"].unique())
    return dct


# ────────────────────────────── 主函数 ──────────────────────────────
def animate_status_over_time_dynamic(
    edge_pattern: str,
    migrated_pattern: str,
    vacant_pattern: str,
    out_gif: str = "network_status.gif",
    base_size: float = 20.0,
    size_scale: float = 5.0,
    interval_ms: int = 1000,
    bridging_keep_fraction: float = 0.01,
    bonding_keep_fraction:  float = 1.0
):
    # ——— 1. 读取所有月份的边表 ———
    edge_dict = load_edges_by_month(edge_pattern)
    if not edge_dict:
        raise ValueError("❌ 未找到符合 pattern 的边文件。")
    months = sorted(edge_dict)

    # ——— 2. 节点全集（先做 bbox 过滤） ———
    union_centers = {}
    for df in edge_dict.values():
        ctr, _ = build_node_info(df)
        for hid, (lat, lon) in ctr.items():
            if south <= lat <= north and west <= lon <= east:
                union_centers[hid] = (lat, lon)
    if not union_centers:
        raise ValueError("❌ 过滤后没有结点落在指定经纬度范围内。")

    nodes_list = list(union_centers)       # 固定顺序
    nodes_set  = set(nodes_list)
    xs = [union_centers[h][1] for h in nodes_list]   # lon
    ys = [union_centers[h][0] for h in nodes_list]   # lat

    # ——— 3. 状态字典（迁移 / 空置） ———
    mig_dict = load_status_dict(migrated_pattern)
    vac_dict = load_status_dict(vacant_pattern)

    # ——— 4. 画布与图例（节点一次性散点） ———
    fig, ax = plt.subplots(figsize=(10, 8))
    scat = ax.scatter(
        xs, ys,
        s=[base_size]*len(nodes_list),
        c=["navy"]*len(nodes_list),
        edgecolors="black", alpha=0.8, zorder=3
    )
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
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 用来存本帧画出的线
    edge_artists = []

    # ——— 5. 帧更新函数 ———
    def update(frame_idx: int):
        nonlocal edge_artists
        date  = months[frame_idx]
        df_e  = edge_dict[date].copy()

        # —— 抽样保留 bridging / bonding 边 ——
        if bridging_keep_fraction < 1.0 or bonding_keep_fraction < 1.0:
            bri = df_e[df_e['type'] == 0]
            bon = df_e[df_e['type'] == 1]
            if bridging_keep_fraction < 1.0 and len(bri):
                bri = bri.sample(frac=bridging_keep_fraction, random_state=42)
            if bonding_keep_fraction < 1.0 and len(bon):
                bon = bon.sample(frac=bonding_keep_fraction, random_state=42)
            df_e = pd.concat([bri, bon], ignore_index=True)

        # —— 过滤到 bbox 内节点 ——
        df_e = df_e[
            df_e['home_1'].isin(nodes_set) &
            (df_e['home_2'].isna() | df_e['home_2'].isin(nodes_set))
        ].reset_index(drop=True)

        # —— 更新节点大小（按当月设备数） ——
        _, counts_month = build_node_info(df_e)
        sizes = [base_size + size_scale * counts_month.get(h, 1) for h in nodes_list]
        scat.set_sizes(sizes)

        # —— 更新节点颜色（状态） ——
        mig_set = mig_dict.get(date, set())
        vac_set = vac_dict.get(date, set())
        colors = []
        for hid in nodes_list:
            if hid in mig_set:       colors.append("red")
            elif hid in vac_set:     colors.append("orange")
            else:                    colors.append("navy")
        scat.set_facecolors(colors)

        # —— 删除上一帧的线，画新线 ——
        for art in edge_artists:
            art.remove()
        edge_artists.clear()

        for _, row in df_e.iterrows():
            if pd.isna(row['home_2']):   # 跳过单端
                continue
            lat1, lon1 = union_centers[row['home_1']]
            lat2, lon2 = union_centers[row['home_2']]
            col = "crimson" if row['type'] == 0 else "darkgreen"
            ln, = ax.plot([lon1, lon2], [lat1, lat2],
                          color=col, alpha=0.6, lw=1, zorder=2)
            edge_artists.append(ln)

        # —— 标题 —— 
        ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)
        # 返回被 blit 更新的艺术家
        return edge_artists + [scat]

    # ——— 6. 动画 ———
    anim = FuncAnimation(
        fig, update, frames=len(months),
        interval=interval_ms, blit=True, repeat=True
    )

    # ——— 7. 保存 ———
    if out_gif.lower().endswith(".gif"):
        anim.save(out_gif, writer="pillow", dpi=120)
    else:
        anim.save(out_gif, writer="ffmpeg", dpi=120)
    plt.close(fig)
    print(f"✅ Saved animation to {out_gif}")


# ────────────────────────────── 使用示例 ──────────────────────────────
# 在调用之前，**必须**提前设置 south / north / west / east 四个经纬度边界，
# 例如：
# south, north = 24.5, 31.0     # 纬度范围
# west,  east  = -88.0, -79.0   # 经度范围

animate_status_over_time_dynamic(
    edge_pattern   = "florida_results/Household_social_network_*.csv",
    migrated_pattern = "florida_results/Migrated_home*.csv",
    vacant_pattern   = "florida_results/Vacant_home*.csv",
    out_gif        = "florida_results/network_status.gif",
    base_size      = 30,
    size_scale     = 10,
    interval_ms    = 1000
)
