import pickle
import pandas as pd

def debug_generation_order(pkl_path, data_folder):
    """
    调试重新生成后的数据顺序问题
    """
    
    print("=== 调试重新生成的数据顺序 ===")
    
    # 1. 检查PKL文件中的决策类型分布
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    print("1. PKL文件中的决策类型分布:")
    decision_counts = {}
    for entry in pkl_data['detailed_log']:
        t = entry['timestep']
        k = entry['decision_type']
        if t not in decision_counts:
            decision_counts[t] = {0: 0, 1: 0, 2: 0}
        decision_counts[t][k] += 1
    
    for t in sorted(decision_counts.keys())[:5]:  # 显示前5个时间步
        counts = decision_counts[t]
        print(f"  t={t}: decision_type 0={counts[0]}, 1={counts[1]}, 2={counts[2]}")
    
    # 2. 检查实际的状态数据
    house_states = pd.read_csv(f"{data_folder}/household_states_raw_with_log2.csv")
    
    print("\n2. 实际状态数据分布:")
    state_cols = ['vacancy_state', 'repair_state', 'sales_state']
    
    for t in [1, 2, 3]:  # 检查前几个时间步
        states_t = house_states[house_states['time'] == t]
        print(f"  t={t}:")
        for i, col in enumerate(state_cols):
            active_count = states_t[col].sum()
            inactive_count = (states_t[col] == 0).sum()
            print(f"    {col}: {inactive_count} inactive, {active_count} active")
    
    # 3. 检查代码中的state_cols定义
    print("\n3. 请检查以下文件中的state_cols定义:")
    print("   generate_household_states.py 中应该是:")
    print("   state_cols = ['vacancy_state', 'repair_state', 'sales_state']")
    print("   并且在log_generator_probabilities函数调用时传入的k值对应:")
    print("   k=0 → vacancy_state")
    print("   k=1 → repair_state") 
    print("   k=2 → sales_state")
    
    # 4. 检查main_generator中的循环
    print("\n4. 检查main_generator_ruxiao.py中的循环:")
    print("   for k, k_col in enumerate(state_dims):")
    print("   这里的state_dims应该是:")
    print("   state_dims = ['vacancy_state', 'repair_state', 'sales_state']")
    print("   确保与generate_household_states.py中的state_cols一致!")
    
    # 5. 验证特定时间步的数据
    print("\n5. 验证t=2的具体数据:")
    
    # 从PKL中提取t=2的数据
    pkl_t2 = {}
    for entry in pkl_data['detailed_log']:
        if entry['timestep'] == 2:
            k = entry['decision_type']
            if k not in pkl_t2:
                pkl_t2[k] = []
            pkl_t2[k].append(entry['household_index'])
    
    # 从状态数据中提取t=2的inactive households
    states_t2 = house_states[house_states['time'] == 2]
    state_t2 = {}
    for i, col in enumerate(state_cols):
        inactive_households = states_t2[states_t2[col] == 0].index.tolist()
        # 需要转换为household indices，这里假设index就是household_index
        state_t2[i] = len(inactive_households)
    
    print("  PKL中t=2的记录数量:")
    for k in sorted(pkl_t2.keys()):
        print(f"    decision_type {k}: {len(pkl_t2[k])} households")
    
    print("  状态数据中t=2的inactive数量:")
    for i in range(3):
        col_name = state_cols[i]
        inactive_count = (states_t2[col_name] == 0).sum()
        print(f"    {col_name} (期望对应decision_type {i}): {inactive_count} inactive households")

if __name__ == "__main__":
    pkl_path = "/Users/susangao/Desktop/CLIMA/CODE 4.6/data/syn_data_ruxiao_v2/detailed_generator_probabilities.pkl"
    data_folder = "/Users/susangao/Desktop/CLIMA/CODE 4.6/data/syn_data_ruxiao_v2"
    
    debug_generation_order(pkl_path, data_folder)