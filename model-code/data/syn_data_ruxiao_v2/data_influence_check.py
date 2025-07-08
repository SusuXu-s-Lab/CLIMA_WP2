import pickle
import pandas as pd
import numpy as np

def load_data(pkl_path, data_folder):
    """加载所有需要的数据"""
    # 加载pkl文件
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # 加载CSV文件
    house_states = pd.read_csv(f"{data_folder}/household_states_raw_with_log2.csv")
    ground_truth_links = pd.read_csv(f"{data_folder}/ground_truth_network_raw_with_log2.csv")
    house_features = pd.read_csv(f"{data_folder}/household_features_raw_with_log2.csv")
    
    return pkl_data, house_states, ground_truth_links, house_features

def get_active_neighbors_from_pkl(pkl_data, target_home, target_t, target_k):
    """
    从pkl文件中获取某个节点在指定时间和决策维度的活跃邻居
    """
    detailed_log = pkl_data['detailed_log']
    
    # 找到匹配的记录
    for record in detailed_log:
        if (record['household_home'] == target_home and 
            record['timestep'] == target_t and 
            record['decision_type'] == target_k):
            
            active_neighbors = record['neighbor_influences']
            neighbor_homes = [ni['neighbor_home'] for ni in active_neighbors]
            
            print(f"PKL记录: {target_home} 在 t={target_t}, k={target_k} 的活跃邻居:")
            print(f"  活跃邻居数量: {len(neighbor_homes)}")
            for ni in active_neighbors:
                print(f"    {ni['neighbor_home']}: link_type={ni['link_type']}, influence={ni['influence_prob']:.6f}")
            
            return neighbor_homes
    
    print(f"未找到 {target_home} 在 t={target_t}, k={target_k} 的记录")
    return []

def get_active_neighbors_from_data(house_states, ground_truth_links, 
                                  target_home, target_t, target_k):
    """
    从原始数据文件重新计算某个节点的活跃邻居
    """
    state_cols = ['repair_state', 'vacancy_state', 'sales_state']
    
    # 检查target_home在t时刻的状态
    target_state = house_states[
        (house_states['time'] == target_t) & 
        (house_states['home'] == target_home)
    ]
    
    if len(target_state) == 0:
        print(f"{target_home} 在 t={target_t} 时无状态记录")
        return []
    
    target_current_state = target_state.iloc[0][state_cols[target_k]]
    if target_current_state == 1:
        print(f"{target_home} 在 t={target_t}, k={target_k} 已经是活跃状态")
        return []
    
    # 找到所有与target_home相连的链接（使用geohash直接搜索）
    links_t = ground_truth_links[ground_truth_links['time_step'] == target_t]
    connected_links = links_t[
        (links_t['household_id_1'] == target_home) | 
        (links_t['household_id_2'] == target_home)
    ]
    
    # 获取t时刻所有房屋的状态
    states_t = house_states[house_states['time'] == target_t].set_index('home')
    
    active_neighbors = []
    
    print(f"数据重算: {target_home} 在 t={target_t}, k={target_k} 的活跃邻居:")
    print(f"  找到 {len(connected_links)} 个连接")
    
    for _, link_row in connected_links.iterrows():
        # 确定邻居的geohash
        if link_row['household_id_1'] == target_home:
            neighbor_geohash = link_row['household_id_2']
        else:
            neighbor_geohash = link_row['household_id_1']
        
        # 检查邻居是否存在状态记录
        if neighbor_geohash not in states_t.index:
            print(f"    跳过: {neighbor_geohash} 在t={target_t}无状态记录")
            continue
        
        # 检查邻居在维度k的状态
        neighbor_state = states_t.loc[neighbor_geohash, state_cols[target_k]]
        link_type = link_row['link_type']
        
        print(f"    检查 {neighbor_geohash}: state={neighbor_state}, link_type={link_type}")
        
        if neighbor_state == 1 and link_type > 0:
            active_neighbors.append(neighbor_geohash)
            print(f"      ✓ 活跃邻居: {neighbor_geohash}")
    
    print(f"  活跃邻居数量: {len(active_neighbors)}")
    return active_neighbors

def compare_neighbors(pkl_path, data_folder, target_home, target_t, target_k):
    """
    比较pkl记录和数据重算的活跃邻居
    """
    print("=" * 60)
    print(f"比较节点 {target_home} 在 t={target_t}, k={target_k} 的活跃邻居")
    print("=" * 60)
    
    # 加载数据
    pkl_data, house_states, ground_truth_links, house_features = load_data(pkl_path, data_folder)
    
    # 从pkl获取
    print("\n【1. 从PKL文件读取】")
    pkl_neighbors = get_active_neighbors_from_pkl(pkl_data, target_home, target_t, target_k)
    
    print("\n【2. 从数据文件重算】")
    data_neighbors = get_active_neighbors_from_data(
        house_states, ground_truth_links, target_home, target_t, target_k
    )
    
    # 比较结果
    print("\n【3. 比较结果】")
    pkl_set = set(pkl_neighbors)
    data_set = set(data_neighbors)
    
    print(f"PKL记录的邻居: {pkl_set}")
    print(f"数据重算邻居: {data_set}")
    print(f"是否一致: {pkl_set == data_set}")
    
    if pkl_set != data_set:
        print(f"PKL多出的邻居: {pkl_set - data_set}")
        print(f"PKL缺少的邻居: {data_set - pkl_set}")
    
    return pkl_neighbors, data_neighbors

# 使用示例
if __name__ == "__main__":
    pkl_path = "/Users/susangao/Desktop/CLIMA/CODE 4.6/data/syn_data_ruxiao_v2/detailed_generator_probabilities.pkl"
    data_folder = "/Users/susangao/Desktop/CLIMA/CODE 4.6/data/syn_data_ruxiao_v2"
    
    # 示例：检查某个具体节点
    target_home = "dqcne6ct"  # 你想检查的geohash
    target_t = 4             # 时间步
    target_k = 0             # 决策维度 (0=repair, 1=vacancy, 2=sales)
    
    pkl_neighbors, data_neighbors = compare_neighbors(pkl_path, data_folder, target_home, target_t, target_k)