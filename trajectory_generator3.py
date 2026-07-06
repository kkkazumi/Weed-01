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

    time_steps = np.arange(0, duration, 1.0 / sampling_rate)
    dt = 1.0 / sampling_rate

    # 左右の手先の初期位置設定 (メートル単位)
    y_left = np.full_like(time_steps, -0.15)
    z_left = np.full_like(time_steps, 0.2)
    y_right = np.full_like(time_steps, 0.15)
    z_right = np.full_like(time_steps, 0.2)

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
            if start_idx >= len(time_steps):
                continue
            if end_idx > len(time_steps):
                end_idx = len(time_steps)

            # 【マッピング調整】実機アーム(合計27cm)が絶対に無理なく届く「安全演奏エリア」にスケール
            # 音高が高くなるほど上（Zプラス）、低くなると下へ
            target_z = 0.08 + (note.pitch - 36) / (96 - 36) * 0.12  # 8cm 〜 20cm
            target_z = np.clip(target_z, 0.08, 0.20)

            # 音量が大きくなるほど、外側（Yの遠く）へアームを伸ばす
            spread = (note.velocity / 127.0) * 0.07  # 広がり幅を最大7cmに制限
            target_y_left = -0.12 - spread  # -12cm 〜 -19cm
            target_y_right = 0.12 + spread  # 10cm 〜 19cm

            # なめらかな直線補間の計算
            n_frames = end_idx - start_idx
            if n_frames > 0:
                # 【バグ修正】Z軸（高さ）の補間には、確実に直前の「Z軸（z_left）」の過去座標を代入します
                start_z_l = z_left[start_idx - 1] if start_idx > 0 else 0.2
                z_left[start_idx:end_idx] = np.linspace(start_z_l, target_z, n_frames)
                z_right[start_idx:end_idx] = np.linspace(start_z_l, target_z, n_frames)

                # Y軸 (左右広がり)
                start_y_l = y_left[start_idx - 1] if start_idx > 0 else -0.15
                start_y_r = y_right[start_idx - 1] if start_idx > 0 else 0.15
                y_left[start_idx:end_idx] = np.linspace(start_y_l, target_y_left, n_frames)
                y_right[start_idx:end_idx] = np.linspace(start_y_r, target_y_right, n_frames)

    # 3. 平滑化
    window_size = int(sampling_rate * 0.25)
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        y_left = np.convolve(y_left, kernel, mode='same')
        y_right = np.convolve(y_right, kernel, mode='same')
        z_left = np.convolve(z_left, kernel, mode='same')
        z_right = np.convolve(z_right, kernel, mode='same')

    # =======================================================
    # 4. FKフィードバック型・二重リミッター処理
    # =======================================================
    MAX_DEG_PER_SEC = 60.0
    MAX_CHANGE_PER_FRAME = MAX_DEG_PER_SEC * dt

    LIMITS_MIN = [-60.0, -120.0, -135.0]
    LIMITS_MAX = [0.0, 120.0, 45.0]

    def get_initial_angles(y_val, z_val):
        # 【修正】引数の phi_deg=0.0 を削除
        ang = inverse_kinematics_3link(float(y_val), float(z_val))
        if ang is not None:
            return [np.clip(ang[j], LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
        return [0.0, 0.0, 0.0]

    prev_angles_l = get_initial_angles(y_left[0], z_left[0])
    prev_angles_r = get_initial_angles(-y_right[0], z_right[0])

    for i in range(1, len(time_steps)):
        # --- 左腕の制限・フィードバック処理 ---
        # 【修正】引数の phi_deg=0.0 を削除
        angles_l = inverse_kinematics_3link(float(y_left[i]), float(z_left[i]))
        if angles_l is not None:
            limited_angles_l = []
            for j in range(3):
                diff = float(angles_l[j]) - float(prev_angles_l[j])
                diff_limited = np.clip(diff, -MAX_CHANGE_PER_FRAME, MAX_CHANGE_PER_FRAME)
                next_ang_clamped = np.clip(prev_angles_l[j] + diff_limited, LIMITS_MIN[j], LIMITS_MAX[j])
                limited_angles_l.append(next_ang_clamped)

            _, _, _, p3_l = forward_kinematics_3link(*limited_angles_l)
            # タプルの1番目(X)をY軸、2番目(Y)をZ軸にマッピング
            y_left[i] = float(p3_l[0]) / 100.0
            z_left[i] = float(p3_l[1]) / 100.0
            prev_angles_l = limited_angles_l
        else:
            y_left[i] = y_left[i - 1]
            z_left[i] = z_left[i - 1]

        # --- 右腕の制限・フィードバック処理 ---
        # 【修正】引数の phi_deg=0.0 を削除
        angles_r = inverse_kinematics_3link(float(-y_right[i]), float(z_right[i]))
        if angles_r is not None:
            limited_angles_r = []
            for j in range(3):
                diff = float(angles_r[j]) - float(prev_angles_r[j])
                diff_limited = np.clip(diff, -MAX_CHANGE_PER_FRAME, MAX_CHANGE_PER_FRAME)
                next_ang_clamped = np.clip(prev_angles_r[j] + diff_limited, LIMITS_MIN[j], LIMITS_MAX[j])
                limited_angles_r.append(next_ang_clamped)

            _, _, _, p3_r = forward_kinematics_3link(*limited_angles_r)
            y_right[i] = float(-p3_r[0]) / 100.0
            z_right[i] = float(p3_r[1]) / 100.0
            prev_angles_r = limited_angles_r
        else:
            y_right[i] = y_right[i - 1]
            z_right[i] = z_right[i - 1]

    return time_steps, y_left, z_left, y_right, z_right, duration


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
            # 【修正】phi_deg=0.0 を完全に消去しました
            angles_l = inverse_kinematics_3link(float(y_l[i]), float(z_l[i]))
            if angles_l is not None:
                th_a_l, th_b_l, th_c_l = [np.clip(float(angles_l[j]), LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
            else:
                th_a_l, th_b_l, th_c_l = -60.0, 120.0, 0.0

            # 【修正】phi_deg=0.0 を完全に消去しました
            angles_r = inverse_kinematics_3link(float(-y_r[i]), float(z_r[i]))
            if angles_r is not None:
                th_a_r, th_b_r, th_c_r = [np.clip(float(angles_r[j]), LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
            else:
                th_a_r, th_b_r, th_c_r = -60.0, 120.0, 0.0

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
