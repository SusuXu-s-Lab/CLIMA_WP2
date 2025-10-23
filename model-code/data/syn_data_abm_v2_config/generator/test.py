from abm_config import TEST2
from abm_data_generator_configurable import ABMDataGenerator
import numpy as np
import pandas as pd

config = TEST2
np.random.seed(config.random_seed)
gen = ABMDataGenerator(config)

print("开始完整模拟...")
data = gen.simulate()

states = data['household_states']

print(f"\n{'='*70}")
print(f"=== 最终结果 ===")
print(f"{'='*70}")

# 按时间点展示激活数
for t in [0, 6, 12, 18, 23]:
    states_t = states[states['timestep'] == t]
    print(f"t={t:2d}: Repair={states_t['repair'].sum():3d}, "
          f"Vacant={states_t['vacant'].sum():3d}, "
          f"Sell={states_t['sell'].sum():3d}")

print(f"\n{'='*70}")
print(f"=== 决策组合统计 (各时间点) ===")
print(f"{'='*70}")

for t in [0, 6, 12, 18, 23]:
    states_t = states[states['timestep'] == t].copy()
    
    # 计算每个household激活的决策总数
    states_t['total_decisions'] = (
        states_t['repair'] + states_t['vacant'] + states_t['sell']
    )
    
    # 统计不同组合
    print(f"\n--- t={t} ---")
    
    # 按激活数量统计
    for n in range(4):  # 0, 1, 2, 3个决策
        count = (states_t['total_decisions'] == n).sum()
        pct = count / len(states_t) * 100
        print(f"  {n}个决策激活: {count:3d} households ({pct:5.1f}%)")
    
    # 详细的组合统计
    print(f"\n  具体组合:")
    
    # 0个决策
    none = ((states_t['repair'] == 0) & 
            (states_t['vacant'] == 0) & 
            (states_t['sell'] == 0)).sum()
    print(f"    (0,0,0): {none:3d} households")
    
    # 1个决策
    repair_only = ((states_t['repair'] == 1) & 
                   (states_t['vacant'] == 0) & 
                   (states_t['sell'] == 0)).sum()
    vacant_only = ((states_t['repair'] == 0) & 
                   (states_t['vacant'] == 1) & 
                   (states_t['sell'] == 0)).sum()
    sell_only = ((states_t['repair'] == 0) & 
                 (states_t['vacant'] == 0) & 
                 (states_t['sell'] == 1)).sum()
    
    print(f"    (1,0,0) Repair only:  {repair_only:3d}")
    print(f"    (0,1,0) Vacant only:  {vacant_only:3d}")
    print(f"    (0,0,1) Sell only:    {sell_only:3d}")
    
    # 2个决策
    repair_vacant = ((states_t['repair'] == 1) & 
                     (states_t['vacant'] == 1) & 
                     (states_t['sell'] == 0)).sum()
    repair_sell = ((states_t['repair'] == 1) & 
                   (states_t['vacant'] == 0) & 
                   (states_t['sell'] == 1)).sum()
    vacant_sell = ((states_t['repair'] == 0) & 
                   (states_t['vacant'] == 1) & 
                   (states_t['sell'] == 1)).sum()
    
    if repair_vacant + repair_sell + vacant_sell > 0:
        print(f"    (1,1,0) Repair+Vacant: {repair_vacant:3d}")
        print(f"    (1,0,1) Repair+Sell:   {repair_sell:3d}")
        print(f"    (0,1,1) Vacant+Sell:   {vacant_sell:3d}")
    
    # 3个决策
    all_three = ((states_t['repair'] == 1) & 
                 (states_t['vacant'] == 1) & 
                 (states_t['sell'] == 1)).sum()
    
    if all_three > 0:
        print(f"    (1,1,1) All three:     {all_three:3d}")

print(f"\n{'='*70}")
print(f"=== 最终状态 (t=23) 汇总 ===")
print(f"{'='*70}")

final_states = states[states['timestep'] == 23].copy()
final_states['total_decisions'] = (
    final_states['repair'] + final_states['vacant'] + final_states['sell']
)

print(f"\n按激活决策数分组:")
decision_counts = final_states['total_decisions'].value_counts().sort_index()
for n_decisions, count in decision_counts.items():
    pct = count / len(final_states) * 100
    print(f"  {n_decisions}个决策: {count:3d} households ({pct:5.1f}%)")

# 如果有2个或3个决策的，展示是哪些household
multi_decision_households = final_states[final_states['total_decisions'] >= 2]
if len(multi_decision_households) > 0:
    print(f"\n多决策households详情 (共{len(multi_decision_households)}个):")
    for _, row in multi_decision_households.iterrows():
        hid = row['household_id']
        r, v, s = row['repair'], row['vacant'], row['sell']
        print(f"  Household {hid:3d}: Repair={r}, Vacant={v}, Sell={s}")