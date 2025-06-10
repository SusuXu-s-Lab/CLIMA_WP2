def update(frame_idx):
    date   = months[frame_idx]
    df_e   = edge_dict[date]

    # …（抽样、build_node_info、bounding-box 过滤等同原来）…

    mig_set = mig_dict.get(date, set())
    vac_set = vac_dict.get(date, set())

    ax.cla()
    ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ── 1️⃣ 先画所有节点 ───────────────────────────────
    xs, ys, sizes, cols = [], [], [], []
    for hid, (lat, lon) in center.items():
        xs.append(lon)
        ys.append(lat)
        sizes.append(base_size + size_scale * counts.get(hid, 1))
        if hid in mig_set:       cols.append("red")
        elif hid in vac_set:     cols.append("orange")
        else:                    cols.append("navy")
    ax.scatter(xs, ys, s=sizes, c=cols,
               edgecolors="black", alpha=0.8, zorder=3)  # 节点层级最高

    # ── 2️⃣ 再在这些节点之间画边 ───────────────────────
    for _, row in df_e.iterrows():
        if pd.isna(row['home_2']):
            continue
        lat1, lon1 = center[row['home_1']]
        lat2, lon2 = center[row['home_2']]
        col = "crimson" if row['type'] == 0 else "darkgreen"
        ax.plot([lon1, lon2], [lat1, lat2],
                color=col, alpha=0.6, lw=1, zorder=1)

    ax.legend(handles=legend_items, loc="upper right")
    return []
