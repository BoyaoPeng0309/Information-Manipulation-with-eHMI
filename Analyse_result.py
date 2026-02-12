import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# =====================
# 数据提取
# =====================
df = pd.read_csv("result/result.csv", low_memory=False)

df_game0 = (
    df[df['game_type'] == 0]
    .groupby('case_id', as_index=False)[['av_payoff', 'hv_payoff']]
    .mean()
)

df_game1 = (
    df[df['game_type'] == 1]
    .groupby('case_id', as_index=False)[['av_payoff', 'hv_payoff']]
    .mean()
)

df_comb = df_game0.merge(
    df_game1,
    on='case_id',
    how='inner',
    suffixes=('_game0', '_game1')
)

scaler = StandardScaler()
df_comb[['av_payoff_game0','av_payoff_game1']] = scaler.fit_transform(df_comb[['av_payoff_game0','av_payoff_game1']])
df_comb[['hv_payoff_game0','hv_payoff_game1']] = scaler.fit_transform(df_comb[['hv_payoff_game0','hv_payoff_game1']])

df_comb['av_payoff_diff'] = (
    df_comb['av_payoff_game0'] - df_comb['av_payoff_game1']
)
df_comb['hv_payoff_diff'] = (
    df_comb['hv_payoff_game0'] - df_comb['hv_payoff_game1']
)

np.random.seed(42)
dup_mask = df_comb.duplicated(
    subset=['av_payoff_diff'],
    keep=False
)
jitter_scale = 0.15
df_comb.loc[dup_mask, 'av_payoff_diff'] += (
    np.random.uniform(-jitter_scale, jitter_scale, size=dup_mask.sum())
)

df_comb['total_payoff_honest'] = (
    df_comb['av_payoff_game1'] + df_comb['hv_payoff_game1']
)

df_comb['total_payoff_deceptive'] = (
    df_comb['av_payoff_game0'] + df_comb['hv_payoff_game0']
)

df_comb.to_csv(
    'df_comb.csv',
    index=False,
    encoding='utf-8-sig'
)

# =====================
# 找出UE变成SO的点
# =====================
total_n = len(df_comb)
df_UE2SO_0 = df_comb[
    df_comb["total_payoff_deceptive"] > df_comb["total_payoff_honest"]
].copy()

df_UE2SO_0['av_payoff_diff_asinh'] = np.arcsinh(df_UE2SO_0['av_payoff_diff']) / 2
df_UE2SO_0['hv_payoff_diff_asinh'] = np.arcsinh(df_UE2SO_0['hv_payoff_diff']) / 2

df_UE2SO = df_UE2SO_0[
    df_UE2SO_0['av_payoff_diff_asinh'] + df_UE2SO_0['hv_payoff_diff_asinh'] > 0
].copy()

n_SO = len(df_UE2SO)

df_non_pareto_type1 = df_UE2SO[
    (df_UE2SO['av_payoff_diff_asinh'] < 0) &
    (df_UE2SO['hv_payoff_diff_asinh'] > 0)
]
n_non_pareto_type1 = len(df_non_pareto_type1)

df_non_pareto_type2 = df_UE2SO[
    (df_UE2SO['av_payoff_diff_asinh'] > 0) &
    (df_UE2SO['hv_payoff_diff_asinh'] < 0)
]
n_non_pareto_type2 = len(df_non_pareto_type2)

n_non_pareto = n_non_pareto_type1 + n_non_pareto_type2

df_pareto = df_UE2SO[
    (df_UE2SO['av_payoff_diff_asinh'] > 0) &
    (df_UE2SO['hv_payoff_diff_asinh'] > 0)
]
n_pareto = len(df_pareto)

df_UE2SO['outcome_type'] = np.select(
    [
        (df_UE2SO['av_payoff_diff_asinh'] > 0) & (df_UE2SO['hv_payoff_diff_asinh'] > 0),
        (df_UE2SO['av_payoff_diff_asinh'] < 0) & (df_UE2SO['hv_payoff_diff_asinh'] > 0),
        (df_UE2SO['av_payoff_diff_asinh'] > 0) & (df_UE2SO['hv_payoff_diff_asinh'] < 0),
    ],
    [
        'pareto',
        'non_pareto_type1',
        'non_pareto_type2'
    ],
    default='other'
)

df_UE2SO.to_csv(
    'df_UE2SO_1.csv',
    index=False,
    encoding='utf-8-sig'
)

# =========================
# 打印结果
# =========================
print("===== Statistics Summary =====")
print(f"SO cases: {n_SO} / {total_n}  ({n_SO/total_n*100:.2f}%)")
print()
print(f"Non-Pareto optimal (Type 1: av < 0, hv > 0): {n_non_pareto_type1} / {n_SO}  ({n_non_pareto_type1/n_SO*100:.2f}%)")
print(f"Non-Pareto optimal (Type 2: av > 0, hv < 0): {n_non_pareto_type2} / {n_SO}  ({n_non_pareto_type2/n_SO*100:.2f}%)")
print(f"Non-Pareto optimal (Total): {n_non_pareto} / {n_SO}  ({n_non_pareto/n_SO*100:.2f}%)")
print()
print(f"Pareto optimal (av > 0, hv > 0): {n_pareto} / {n_SO}  ({n_pareto/n_SO*100:.2f}%)")


# =========================
# 四象限图
# =========================
import matplotlib.pyplot as plt

x = df_UE2SO['av_payoff_diff_asinh']
y = df_UE2SO['hv_payoff_diff_asinh']

plt.figure(figsize=(6, 6))
plt.scatter(x, y, alpha=0.6, s=10)

plt.axhline(0, linestyle='--', linewidth=0.8)
plt.axvline(0, linestyle='--', linewidth=0.8)

plt.xlabel('arcsinh(ΔUA)')
plt.ylabel('arcsinh(ΔUH)')
plt.title('ΔUA vs ΔUH (arcsinh scale)')
plt.gca().set_aspect('equal')
# plt.axis([-1, 1, -1, 1])

plt.tight_layout()
plt.show()
