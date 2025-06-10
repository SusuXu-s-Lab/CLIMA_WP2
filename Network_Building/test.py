def update(frame_idx):
    # ───── 0. 取出当月数据 ─────────────────────────────
    date = months[frame_idx]
    df_e = edge_dict[date]

    # 可选抽样：按需保留部分 bridging / bonding 边
    if bridging_keep_fraction < 1.0 or bonding_keep_fraction < 1.0:
        bri = df_e[df_e['type'] == 0]        # bridging
        bon = df_e[df_e['type'] == 1]        # bonding
        if bridging_keep_fraction < 1.0 and len(bri):
            bri = bri.sample(frac=bridging_keep_fraction,
                             random_state=42)
        if bonding_keep_fraction < 1.0 and len(bon):
            bon = bon.sample(frac=bonding_keep_fraction,
                             random_state=42)
        df_e = pd.concat([bri, bon], ignore_index=True)

    # ───── 1. 生成当月所有节点信息 ─────────────────────
    center, counts = build_node_info(df_e)   # ← 生成 center
    # 可选：bounding-box 过滤（确保已定义 south/north/west/east）
    center = {
        h: (lat, lon)
        for h, (lat, lon) in center.items()
        if south <= lat <= north and west <= lon <= east
    }
    df_e = df_e[
        df_e['home_1'].isin(center) &
        (df_e['home_2'].isna() | df_e['home_2'].isin(center))
    ].reset_index(drop=True)

    # ───── 2. 状态集合 ────────────────────────────────
    mig_set = mig_dict.get(date, set())
    vac_set = vac_dict.get(date, set())

    # ───── 3. 清画布 & 标题 ───────────────────────────
    ax.cla()
    ax.set_title(f"Migration & Vacancy   {date:%Y-%m}", fontsize=18)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # ───── 4. 先画所有节点 ────────────────────────────
    xs, ys, sizes, cols = [], [], [], []
    for hid, (lat, lon) in center.items():
        xs.append(lon)
        ys.append(lat)
        sizes.append(base_size + size_scale * counts.get(hid, 1))
        if hid in mig_set:       cols.append("red")
        elif hid in vac_set:     cols.append("orange")
        else:                    cols.append("navy")
    ax.scatter(xs, ys, s=sizes, c=cols,
               edgecolors="black", alpha=0.8, zorder=3)

    # ───── 5. 再添加边 ────────────────────────────────
    for _, row in df_e.iterrows():
        if pd.isna(row['home_2']):            # 忽略无目标的“边”
            continue
        lat1, lon1 = center[row['home_1']]
        lat2, lon2 = center[row['home_2']]
        col = "crimson" if row['type'] == 0 else "darkgreen"
        ax.plot([lon1, lon2], [lat1, lat2],
                color=col, alpha=0.6, lw=1, zorder=1)

    # ───── 6. 固定图例 ────────────────────────────────
    ax.legend(handles=legend_items, loc="upper right")
    return []
