import pandas as pd
import numpy as np

# =====================
# 1. 读取数据
# =====================
df_trust = pd.read_csv(r'C:\Users\Manan\Desktop\game_final\result\result.csv')
df_trust = df_trust[df_trust['game_type'] == 0]
df_ue2so = pd.read_csv(r'C:\Users\Manan\Desktop\game_final\df_UE2SO_1.csv')

print("=" * 50)
print("原始数据概览")
print("=" * 50)
print(f"trust.xlsx: {df_trust.shape[0]} 行, {df_trust['case_id'].nunique()} 个case")
print(f"df_UE2SO.csv: {df_ue2so.shape[0]} 行, {df_ue2so['case_id'].nunique()} 个case")

# =====================
# 2. 筛选系统最优案例
# =====================
# 只保留df_UE2SO中存在的case_id
valid_case_ids = df_ue2so['case_id'].unique()
df_filtered = df_trust[df_trust['case_id'].isin(valid_case_ids)].copy()

# 合并outcome_type
df_filtered = df_filtered.merge(
    df_ue2so[['case_id', 'outcome_type']], 
    on='case_id', 
    how='left'
)

before_rows = df_filtered.shape[0]

df_filtered = df_filtered[
    ~(
        (df_filtered['av_estimated_trust'] == 0.5) &
        (df_filtered['av_estimated_trust_std'] == 0.2)
    )
].copy()

df_filtered = df_filtered[
    df_filtered.groupby('case_id')['bf']
               .transform(lambda x: not (x == 1).all())
].copy()

after_rows = df_filtered.shape[0]

print(f"\n去除 av_estimated_trust=0.5 且 av_estimated_trust_std=0.2 后：")
print(f"删除行数: {before_rows - after_rows}")
print(f"剩余行数: {after_rows}")
print(f"剩余 case 数: {df_filtered['case_id'].nunique()}")

print(f"\n筛选后: {df_filtered.shape[0]} 行, {df_filtered['case_id'].nunique()} 个case")
print(f"outcome_type分布:\n{df_filtered.groupby('case_id')['outcome_type'].first().value_counts()}")

# =====================
# 3. 判断欺骗成功/失败
# =====================
df_filtered = df_filtered[
    ~(
        (df_filtered['deceptive'] == 'None')
    )
].copy()

df_filtered['deception_success'] = (df_filtered['deceptive'] == 'success').astype(int)

# =====================
# 4. 分析1：欺骗成功率随信任度的变化（按outcome_type分类）
# =====================
print("\n" + "=" * 50)
print("分析1：欺骗成功率随信任度变化")
print("=" * 50)

# 对bf进行分箱
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
df_filtered['bf_bin'] = pd.cut(df_filtered['bf'], bins=bins, labels=bin_labels, include_lowest=True)

# 按outcome_type和bf_bin计算成功率
success_rate_by_type = df_filtered.groupby(['outcome_type', 'bf_bin']).agg(
    total_count=('deception_success', 'count'),
    success_count=('deception_success', 'sum'),
    success_rate=('deception_success', 'mean')
).reset_index()

print(success_rate_by_type)

# 整体成功率（不分类型）
success_rate_overall = df_filtered.groupby('bf_bin').agg(
    total_count=('deception_success', 'count'),
    success_count=('deception_success', 'sum'),
    success_rate=('deception_success', 'mean')
).reset_index()
success_rate_overall['outcome_type'] = 'overall'

print("\n整体成功率:")
print(success_rate_overall)

# 合并结果
success_rate_all = pd.concat([success_rate_by_type, success_rate_overall], ignore_index=True)

# =====================
# 5. 分析2：每个案例的bf均值（信任度演化）
# =====================
print("\n" + "=" * 50)
print("分析2：每个案例的信任度统计")
print("=" * 50)

# 按case_id计算bf的统计量
case_stats = df_filtered.groupby('case_id').agg(
    bf_mean=('bf', 'mean'),
    bf_std=('bf', 'std'),
    bf_min=('bf', 'min'),
    bf_max=('bf', 'max'),
    bf_start=('bf', 'first'),
    bf_end=('bf', 'last'),
    tau_mean=('tau_threshold', 'mean'),
    tau_max=('tau_threshold', 'max'),
    total_steps=('bf', 'count'),
    success_count=('deception_success', 'sum'),
    success_rate=('deception_success', 'mean'),
    outcome_type=('outcome_type', 'first')
).reset_index()

# 计算信任度变化
case_stats['bf_change'] = case_stats['bf_end'] - case_stats['bf_start']
case_stats['bf_change_pct'] = (case_stats['bf_change'] / case_stats['bf_start']) * 100

print(case_stats.head(10))

# 按outcome_type统计
type_summary = case_stats.groupby('outcome_type').agg(
    case_count=('case_id', 'count'),
    bf_mean_avg=('bf_mean', 'mean'),
    bf_mean_std=('bf_mean', 'std'),
    bf_change_avg=('bf_change', 'mean'),
    bf_change_std=('bf_change', 'std'),
    success_rate_avg=('success_rate', 'mean'),
    success_rate_std=('success_rate', 'std')
).reset_index()

print("\n按outcome_type汇总:")
print(type_summary)

# =====================
# 6. 分析3：信任度时序演化（用于绘制时序图）
# =====================
print("\n" + "=" * 50)
print("分析3：信任度时序演化数据")
print("=" * 50)

# 按时间步骤和outcome_type统计
# 首先需要标准化时间步骤（每个case内部的相对时间）
df_filtered['step'] = df_filtered.groupby('case_id').cumcount()

time_evolution = df_filtered.groupby(['outcome_type', 'step']).agg(
    bf_mean=('bf', 'mean'),
    bf_std=('bf', 'std'),
    bf_count=('bf', 'count'),
    tau_mean=('tau_threshold', 'mean'),
    success_rate=('deception_success', 'mean')
).reset_index()

# 计算95%置信区间
time_evolution['bf_se'] = time_evolution['bf_std'] / np.sqrt(time_evolution['bf_count'])
time_evolution['bf_ci_lower'] = time_evolution['bf_mean'] - 1.96 * time_evolution['bf_se']
time_evolution['bf_ci_upper'] = time_evolution['bf_mean'] + 1.96 * time_evolution['bf_se']

print(time_evolution.head(20))

# =====================
# 7. 分析4：Gap分析（bf - tau_threshold）
# =====================
print("\n" + "=" * 50)
print("分析4：Gap分析 (bf - tau_threshold)")
print("=" * 50)

df_filtered['gap'] = df_filtered['bf'] - df_filtered['tau_threshold']

gap_by_type = df_filtered.groupby('outcome_type').agg(
    gap_mean=('gap', 'mean'),
    gap_std=('gap', 'std'),
    gap_min=('gap', 'min'),
    gap_max=('gap', 'max'),
    gap_median=('gap', 'median')
).reset_index()

print(gap_by_type)

# Gap分布数据
gap_distribution = df_filtered.groupby(['outcome_type', pd.cut(df_filtered['gap'], bins=20)]).size().reset_index(name='count')
gap_distribution.columns = ['outcome_type', 'gap_bin', 'count']

# =====================
# 8. 保存结果
# =====================
output_dir = r'C:\Users\Manan\Desktop\game_final\outputs'

# 保存各个分析结果
success_rate_all.to_csv(f'{output_dir}/analysis1_success_rate_by_bf_bin.csv', index=False)
case_stats.to_csv(f'{output_dir}/analysis2_case_statistics.csv', index=False)
type_summary.to_csv(f'{output_dir}/analysis2_type_summary.csv', index=False)
time_evolution.to_csv(f'{output_dir}/analysis3_time_evolution.csv', index=False)
gap_by_type.to_csv(f'{output_dir}/analysis4_gap_summary.csv', index=False)

# 为Origin绘图准备的宽格式数据
# 成功率数据转换为宽格式
success_rate_wide = success_rate_by_type.pivot(
    index='bf_bin', 
    columns='outcome_type', 
    values='success_rate'
).reset_index()
success_rate_wide.columns.name = None
success_rate_wide.to_csv(f'{output_dir}/origin_success_rate_wide.csv', index=False)

# 时序数据转换为宽格式
time_evo_bf = time_evolution.pivot(
    index='step',
    columns='outcome_type',
    values='bf_mean'
).reset_index()
time_evo_bf.columns.name = None
time_evo_bf.to_csv(f'{output_dir}/origin_time_evolution_bf_wide.csv', index=False)

time_evo_success = time_evolution.pivot(
    index='step',
    columns='outcome_type',
    values='success_rate'
).reset_index()
time_evo_success.columns.name = None
time_evo_success.to_csv(f'{output_dir}/origin_time_evolution_success_wide.csv', index=False)

print("\n" + "=" * 50)
print("输出文件列表")
print("=" * 50)
print("1. analysis1_success_rate_by_bf_bin.csv - 欺骗成功率按信任度分箱")
print("2. analysis2_case_statistics.csv - 每个案例的详细统计")
print("3. analysis2_type_summary.csv - 按outcome_type的汇总统计")
print("4. analysis3_time_evolution.csv - 信任度时序演化数据")
print("5. analysis4_gap_summary.csv - Gap分析汇总")
print("6. origin_success_rate_wide.csv - Origin绘图用：成功率宽格式")
print("7. origin_time_evolution_bf_wide.csv - Origin绘图用：信任度演化宽格式")
print("8. origin_time_evolution_success_wide.csv - Origin绘图用：成功率演化宽格式")

print("\n分析完成！")