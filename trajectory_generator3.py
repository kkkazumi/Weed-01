import csv
import numpy as np
import pretty_midi
from kinematics2 import inverse_kinematics_3link, forward_kinematics_3link


def generate_trajectory(midi_path, start_bar=0, num_bars=None, sampling_rate=10):
    """
    MIDI から左右の手先軌道を生成し、
    YZ平面に対して完全に左右対称になるようにマッピングした上で、サーボ制限と可動範囲制限をかける。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    # 1. テンポ(BPM)と 1 小節あたりの時間を計算
    tempo_times, tempo_bpms = pm.get_tempo_changes()

    # 【超重要修正】配列に複数のテンポ（BPM）が入っていても、先頭の要素[0]を指定して安全に数値化します
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

    # 左右の手先の初期位置設定 (メートル単位)
    y_left_ref = np.full_like(base_time_steps, -0.15)
    z_left_ref = np.full_like(base_time_steps, 0.2)
    y_right_ref = np.full_like(base_time_steps, 0.15)
    z_right_ref = np.full_like(base_time_steps, 0.2)

    # 2. MIDI ノートのスキャンとマッピング
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

            # 可動範囲(theta_a: -60〜0度)に収まる安全な目標位置 [m]
            target_z = 0.08 + (note.pitch - 36) / (96 - 36) * 0.12  # 8cm 〜 20cm
            target_z = np.clip(target_z, 0.08, 0.20)

            spread = (note.velocity / 127.0) * 0.07  # 最大7cmに制限
            target_y_left = -0.12 - spread  # -12cm 〜 -19cm
            target_y_right = 0.12 + spread  # 12cm 〜 19cm

            z_left_ref[start_idx:end_idx] = target_z
            z_right_ref[start_idx:end_idx] = target_z
            y_left_ref[start_idx:end_idx] = target_y_left
            y_right_ref[start_idx:end_idx] = target_y_right

    # =======================================================
    # 3. 論文式(4)に基づく時間軸の引き延ばし ＆ 地続き線形補間
    # =======================================================
    MAX_DEG_PER_SEC = 60.0  # 実機の速度上限 [deg/s]
    MAX_CHANGE_PER_FRAME = MAX_DEG_PER_SEC * dt

    LIMITS_MIN = [-60.0, -120.0, -135.0]
    LIMITS_MAX = [0.0, 120.0, 45.0]

    def get_angles(y_val, z_val, is_right=False):
        y_in = -float(y_val) if is_right else float(y_val)
        ang = inverse_kinematics_3link(y_in, float(z_val))
        if ang is not None:
            return [np.clip(ang[j], LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
        return [-30.0, -60.0, 35.0]

    # 【根本解決】IndexErrorを防ぐため、1コマ目の初期値をあらかじめリストに入れて初期化します
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

        max_diff = 0.0
        for j in range(3):
            max_diff = max(max_diff, abs(next_ang_l[j] - prev_ang_l[j]), abs(next_ang_r[j] - prev_ang_r[j]))

        # 論文式(4): 最小必要時間のフレーム数算出
        needed_frames = int(np.ceil(max_diff / MAX_CHANGE_PER_FRAME))
        needed_frames = max(1, needed_frames)  # 最低1フレーム

        step_dt = needed_frames * dt

        # 時間軸の更新
        new_time = optimized_times[-1] + step_dt
        optimized_times.append(new_time)

        # 【バグ完全排除】リストの最後の数値を安全に取り出し、1コマずつ綺麗な坂道(linspace)で隙間を補間
        last_y_l = optimized_y_l[-1][-1] if hasattr(optimized_y_l[-1], "__len__") else optimized_y_l[-1]
        last_z_l = optimized_z_l[-1][-1] if hasattr(optimized_z_l[-1], "__len__") else optimized_z_l[-1]
        last_y_r = optimized_y_r[-1][-1] if hasattr(optimized_y_r[-1], "__len__") else optimized_y_r[-1]
        last_z_r = optimized_z_r[-1][-1] if hasattr(optimized_z_r[-1], "__len__") else optimized_z_r[-1]

        optimized_y_l.append(np.linspace(last_y_l, y_left_ref[i], needed_frames))
        optimized_z_l.append(np.linspace(last_z_l, z_left_ref[i], needed_frames))
        optimized_y_r.append(np.linspace(last_y_r, y_right_ref[i], needed_frames))
        optimized_z_r.append(np.linspace(last_z_r, z_right_ref[i], needed_frames))

        prev_ang_l = next_ang_l
        prev_ang_r = next_ang_r

    # 各アームの配列の平坦化 (入れ子リストの解除)
    def flatten_list(nested_list):
        flat = []
        for item in nested_list:
            if hasattr(item, "__len__"):
                flat.extend(item)
            else:
                flat.append(item)
        return np.array(flat)

    flat_y_l = flatten_list(optimized_y_l)
    flat_z_l = flatten_list(optimized_z_l)
    flat_y_r = flatten_list(optimized_y_r)
    flat_z_r = flatten_list(optimized_z_r)

    # 拡張した総時間に合わせた新しいリサンプリング時間軸の生成
    total_duration = optimized_times[-1]
    total_frames = len(flat_y_l)
    final_time_steps = np.linspace(0, total_duration, total_frames)

    # 4. なだらかなS字躍動スプライン(Minimum Jerk)を適用
    window_size = int(sampling_rate * 0.40)
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        y_l_scaled = np.convolve(flat_y_l, kernel, mode='same')
        z_l_scaled = np.convolve(flat_z_l, kernel, mode='same')
        y_r_scaled = np.convolve(flat_y_r, kernel, mode='same')
        z_r_scaled = np.convolve(flat_z_r, kernel, mode='same')
    else:
        y_l_scaled, z_l_scaled = flat_y_l, flat_z_l
        y_r_scaled, z_r_scaled = flat_y_r, flat_z_r

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
