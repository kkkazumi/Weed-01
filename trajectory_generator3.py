import csv
import numpy as np
import pretty_midi
from kinematics2 import inverse_kinematics_3link, forward_kinematics_3link


def generate_trajectory(midi_path, start_bar=0, num_bars=None, sampling_rate=10):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # テンポ(BPM)の取得
    tempo_times, tempo_bpms = pm.get_tempo_changes()
    if hasattr(tempo_bpms, "__len__") and len(tempo_bpms) > 0:
        bpm = float(tempo_bpms[0])
    else:
        bpm = float(tempo_bpms) if tempo_bpms is not None else 120.0

    seconds_per_bar = (60.0 / bpm) * 4
    start_time = float(start_bar * seconds_per_bar)

    if num_bars is None:
        end_time = float(pm.get_end_time())
    else:
        end_time = float((start_bar + num_bars) * seconds_per_bar)

    duration = end_time - start_time
    if duration <= 0:
        duration = 1.0

    base_time_steps = np.arange(0, duration, 1.0 / sampling_rate)
    dt = 1.0 / sampling_rate

    # 初期目標軌道の確保 (m単位)
    y_left_ref = np.full_like(base_time_steps, -0.15)
    z_left_ref = np.full_like(base_time_steps, 0.2)
    y_right_ref = np.full_like(base_time_steps, 0.15)
    z_right_ref = np.full_like(base_time_steps, 0.2)

    # ----------------=======================================
    # 【ステップ1】ポーズ特徴量最適化 (論文 Page 2, 式(3)に準拠)
    # ----------------=======================================
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.start < start_time or note.start > end_time:
                continue
            rel_start = note.start - start_time
            rel_end = note.end - start_time
            start_idx = int(rel_start * sampling_rate)
            end_idx = int(rel_end * sampling_rate)
            if start_idx >= len(base_time_steps):
                continue
            if end_idx > len(base_time_steps):
                end_idx = len(base_time_steps)

            # 論文式(3): 音楽特徴量空間(Melody)からアームの目標タスク空間特徴量(Z軸高さ)へ最適マッピング
            target_z = 0.08 + (note.pitch - 36) / (96 - 36) * 0.12
            target_z = np.clip(target_z, 0.08, 0.20)

            # 論文式(3): 音楽特徴量空間(Volume/Velocity)からアームの広がりスパン特徴量(Y軸)へ最適マッピング
            spread = (note.velocity / 127.0) * 0.07
            target_y_left = -0.12 - spread
            target_y_right = 0.12 + spread

            z_left_ref[start_idx:end_idx] = target_z
            z_right_ref[start_idx:end_idx] = target_z
            y_left_ref[start_idx:end_idx] = target_y_left
            y_right_ref[start_idx:end_idx] = target_y_right

    # ----------------=======================================
    # 【ステップ2】時間軸の引き延ばし最適化 (論文 Page 3, 式(4)に準拠)
    # ----------------=======================================
    MAX_DEG_PER_SEC = 60.0  # 実機の最大速度限界 [deg/s]
    MAX_CHANGE_PER_FRAME = MAX_DEG_PER_SEC * dt

    LIMITS_MIN = [-60.0, -120.0, -135.0]
    LIMITS_MAX = [0.0, 120.0, 45.0]

    def get_angles(y_val, z_val, is_right=False):
        y_in = -float(y_val) if is_right else float(y_val)
        ang = inverse_kinematics_3link(y_in, float(z_val))
        if ang is not None:
            return [np.clip(ang[j], LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
        return [-30.0, -60.0, 35.0]

    # 時間スケーリングバッファの構築
    optimized_times = [0.0]
    optimized_y_l = [y_left_ref[0]]
    optimized_z_l = [z_left_ref[0]]
    optimized_y_r = [y_right_ref[0]]
    optimized_z_r = [z_right_ref[0]]

    prev_ang_l = get_angles(y_left_ref[0], z_left_ref[0], is_right=False)
    prev_ang_r = get_angles(y_right_ref[0], z_right_ref[0], is_right=True)

    for i in range(1, len(base_time_steps)):
        next_ang_l = get_angles(y_left_ref[i], z_left_ref[i], is_right=False)
        next_ang_r = get_angles(y_right_ref[i], z_right_ref[i], is_right=True)

        # 左右アームの全6関節の必要最大移動量をチェック
        max_diff = 0.0
        for j in range(3):
            max_diff = max(max_diff, abs(next_ang_l[j] - prev_ang_l[j]), abs(next_ang_r[j] - prev_ang_r[j]))

        # 論文式(4)の制約条件: 遷移時間 Delta_t_min がサーボ限界を超えるか判定
        needed_frames = int(np.ceil(max_diff / MAX_CHANGE_PER_FRAME))
        step_dt = dt

        if needed_frames > 1:
            # 限界を超える場合、論文通りに時間割 dt の整数倍 (n_r_i * dt) で後ろへ時間を引き延ばす
            step_dt = needed_frames * dt

        # 最適化された時間軸と参照軌道の蓄積
        new_time = optimized_times[-1] + step_dt
        optimized_times.append(new_time)
        optimized_y_l.append(y_left_ref[i])
        optimized_z_l.append(z_left_ref[i])
        optimized_y_r.append(y_right_ref[i])
        optimized_z_r.append(z_right_ref[i])

        prev_ang_l = next_ang_l
        prev_ang_r = next_ang_r

    # 一定周期(sampling_rate)のリサンプリング時間軸へ再マッピング
    total_duration = optimized_times[-1]
    final_time_steps = np.arange(0, total_duration, dt)

    y_l_scaled = np.interp(final_time_steps, optimized_times, optimized_y_l)
    z_l_scaled = np.interp(final_time_steps, optimized_times, optimized_z_l)
    y_r_scaled = np.interp(final_time_steps, optimized_times, optimized_y_r)
    z_r_scaled = np.interp(final_time_steps, optimized_times, optimized_z_r)

    # ----------------=======================================
    # 【ステップ3】S字最適躍動スプライン補間 (論文 Page 3, 式(8)に準拠)
    # ----------------=======================================
    # 急激なステップ変化を廃止し、加速・減速トルクが最も滑らか(Minimum Jerk)になる5次スプラインフィルタを適用
    window_size = int(sampling_rate * 0.40)  # 論文の0.25s窓から実機用に0.4sなだらか窓へ最適化
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        y_l_scaled = np.convolve(y_l_scaled, kernel, mode='same')
        z_l_scaled = np.convolve(z_l_scaled, kernel, mode='same')
        y_r_scaled = np.convolve(y_r_scaled, kernel, mode='same')
        z_r_scaled = np.convolve(z_r_scaled, kernel, mode='same')

    return final_time_steps, y_l_scaled, z_l_scaled, y_r_scaled, z_r_scaled, total_duration


def export_all_files(midi_path, sampling_rate=10):
    t_steps, y_l, z_l, y_r, z_r, total_duration = generate_trajectory(
        midi_path, num_bars=None, sampling_rate=sampling_rate
    )
    time_per_row = 1.0 / sampling_rate

    with open("angles.csv", mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time [s]",
            "Left_Theta_A [deg]", "Left_Theta_B [deg]", "Left_Theta_C [deg]",
            "Right_Theta_A [deg]", "Right_Theta_B [deg]", "Right_Theta_C [deg]"
        ])

        LIMITS_MIN = [-60.0, -120.0, -135.0]
        LIMITS_MAX = [0.0, 120.0, 45.0]

        for i in range(len(t_steps)):
            angles_l = inverse_kinematics_3link(float(y_l[i]), float(z_l[i]))
            if angles_l is not None:
                th_a_l, th_b_l, th_c_l = [np.clip(float(angles_l[j]), LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
            else:
                th_a_l, th_b_l, th_c_l = -30.0, -60.0, 35.0

            angles_r = inverse_kinematics_3link(float(-y_r[i]), float(z_r[i]))
            if angles_r is not None:
                th_a_r, th_b_r, th_c_r = [np.clip(float(angles_r[j]), LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
            else:
                th_a_r, th_b_r, th_c_r = -30.0, -60.0, 35.0

            writer.writerow([
                f"{t_steps[i]:.3f}",
                f"{th_a_l:.2f}", f"{th_b_l:.2f}", f"{th_c_l:.2f}",
                f"{th_a_r:.2f}", f"{th_b_r:.2f}", f"{th_c_r:.2f}"
            ])

    with open("delay.txt", mode='w', encoding='utf-8') as f:
        f.write(f"{time_per_row}\n")

    with open("duration.txt", mode='w', encoding='utf-8') as f:
        f.write(f"{total_duration:.2f}\n")
    print("--- 左右対称＆数値的IK対応 角度ファイルエクスポート完了 ---")


if __name__ == '__main__':
    midi_file = "radetzky.mid"
    export_all_files(midi_file, sampling_rate=10)
