# main.py
import copy
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from modules.simulator import simulate

def main():

    print("Begin Task...")

    # =====================
    # 仿真参数
    # =====================
    dt = 0.1
    y_left = 3.5
    y_right = 0.0

    # =====================
    # 读取案例和采样
    # =====================
    raw = np.load("initial_distributions.npz")
    speed_samples = raw["speed_diffs"]
    position_samples = raw["position_diffs"]
    speed_samples = speed_samples[(speed_samples >= -5) & (speed_samples <= 20)]
    position_samples = position_samples[(position_samples >= 5) & (position_samples <= 50)]
    num_bins_speed = min(30, max(10, len(speed_samples)//50))
    num_bins_pos = min(30, max(10, len(position_samples)//50))

    speed_hist, speed_bin_edges = np.histogram(speed_samples, bins=num_bins_speed, density=True)
    speed_bin_probs = speed_hist * np.diff(speed_bin_edges)
    speed_bin_probs /= speed_bin_probs.sum()
    speed_bin_centers = (speed_bin_edges[:-1] + speed_bin_edges[1:]) / 2

    pos_hist, pos_bin_edges = np.histogram(position_samples, bins=num_bins_pos, density=True)
    pos_bin_probs = pos_hist * np.diff(pos_bin_edges)
    pos_bin_probs /= pos_bin_probs.sum()
    pos_bin_centers = (pos_bin_edges[:-1] + pos_bin_edges[1:]) / 2

    N_SAMPLES = 1168
    print(f"Total {N_SAMPLES} Cases.")

    sampled_speed_diffs = np.random.choice(speed_bin_centers, size=N_SAMPLES, p=speed_bin_probs)
    sampled_position_diffs = np.random.choice(pos_bin_centers, size=N_SAMPLES, p=pos_bin_probs)


    # =====================
    # 初始状态设定
    # =====================
    HV_V0 = 20.0   # 假设 HV 初始速度固定为 20
    HV_X0 = 20.0   # 假设 HV 初始位置固定为 20

    cases_df = pd.DataFrame({
        "init_spacing": sampled_position_diffs,
        "vel_diff": np.abs(sampled_speed_diffs),
        "targetf_initial_vx": HV_V0,
    })

    cases_df["ego_initial_vx"] = HV_V0 - sampled_speed_diffs
    cases_df["ego_initial_x"] = HV_X0 + sampled_position_diffs
    cases_df["targetf_initial_vx"] = HV_V0
    cases_df["targetf_initial_x"] = HV_X0


    # =====================
    # 批量实验参数设置
    # =====================
    belief_value = 1
    game_types = [0, 1]


    # =====================
    # 主循环：遍历每个案例
    # =====================
    total_sims = len(cases_df) * len(game_types)
    print(f'Total {total_sims} Sim Steps.')
    cnt = 1
    result_list = []

    for idx, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Cases"):
        av_x = row["ego_initial_x"]
        av_v = row["ego_initial_vx"]
        hvs_x = row["targetf_initial_x"]
        hvs_v = row["targetf_initial_vx"]

        for game_type in game_types:
            
            # 运行仿真
            history = simulate(
                {"x": av_x, "v": av_v},     # AV初始状态
                {"x": hvs_x, "v": hvs_v},   # HV初始状态
                y_left,                     # 左侧车道线
                y_right,                    # 右侧车道线
                belief_value,               # 初始信任度
                game_type,                  # 游戏模式
                dt=dt,                      # 仿真精度
            )

            # 保存数据
            df0 = pd.DataFrame(history)
            df0["case_id"] = idx
            df0["game_type"] = game_type
            result_list.append(df0)

            cnt += 1

    # =====================
    # 文件保存
    # =====================
    # 根据 game_type 创建文件夹
    result_dir0 = "result"
    os.makedirs(result_dir0, exist_ok=True)
    filename0 = (f"result.csv")
    file_path0 = os.path.join(result_dir0, filename0)
    all_results = pd.concat(result_list, ignore_index=True)
    all_results.to_csv(file_path0, index=False, encoding="utf-8-sig")
    print("\nTask Finish.")



if __name__ == "__main__":
    main()
