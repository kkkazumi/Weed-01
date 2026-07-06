import csv
import numpy as np
import pretty_midi

# kinematics から正しい関数をインポート
from kinematics import inverse_kinematics_3link, forward_kinematics_3link


def generate_trajectory(midi_path, start_bar=0, num_bars=None, sampling_rate=10):
    """
    MIDI から左右の手先軌道を生成し、サーボ制限と可動範囲制限をかける。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    # 1. テンポ(BPM)と 1 小節あたりの時間を計算
    tempo_times, tempo_bpms = pm.get_tempo_changes()

    # 【超重要修正】配列の一番最初の要素 [0] から、本来の正しいBPM数値を正確に取り出す
    if hasattr(tempo_bpms, "__len__") and len(tempo_bpms) > 0:
        bpm = float(tempo_bpms[0])  # [0] を指定して配列の先頭の値を取得
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

            target_z = 0.1 + (note.pitch - 36) / (96 - 36) * 0.35
            target_z = np.clip(target_z, 0.1, 0.45)

            spread = (note.velocity / 127.0) * 0.25
            target_y_left = -0.05 - spread
            target_y_right = 0.05 + spread

            # 【大改良】固定値代入から、音符の長さをフルに使ったなめらかな直線補間（スライド）に変更
            n_frames = end_idx - start_idx
            if n_frames > 0:
                # 1. Z軸：直前の位置から、今回の目標Zまでゆっくり移動させる
                start_z_l = y_left[start_idx - 1] if start_idx > 0 else 0.2
                z_left[start_idx:end_idx] = np.linspace(start_z_l, target_z, n_frames)
                z_right[start_idx:end_idx] = np.linspace(start_z_l, target_z, n_frames)  # 右も同期

                # 2. Y軸：直前の広がり位置から、今回の目標Yまでゆっくり移動させる
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
    # 4. 【大改良】FKフィードバック型・二重リミッター処理
    # =======================================================
    MAX_DEG_PER_SEC = 60.0  # 60度/秒
    MAX_CHANGE_PER_FRAME = MAX_DEG_PER_SEC * dt  # 1フレームで動いていい最大角度 [deg]

    LIMITS_MIN = [-60.0, -120.0, -135.0]
    LIMITS_MAX = [0.0, 120.0, 45.0]

    def get_initial_angles(y_val, z_val):
        ang = inverse_kinematics_3link(float(y_val), float(z_val), phi_deg=0.0)
        if ang is not None:
            return [np.clip(ang[j], LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
        return [0.0, 0.0, 0.0]

    # 1つ前の「実際に確定した」安全な角度
    prev_angles_l = get_initial_angles(y_left[0], z_left[0])
    prev_angles_r = get_initial_angles(y_right[0], z_right[0])

    for i in range(1, len(time_steps)):
        # ---------------------------------------------------
        # 【左腕】FKフィードバック追従ループ
        # ---------------------------------------------------
        # 前のフレームで「実際にアームが到達した修正済みの座標」をベースにIKを解く
        # (これにより、目標がワープしてもアームは現在地から地続きで計算を始められます)
        angles_l = inverse_kinematics_3link(float(y_left[i]), float(z_left[i]), phi_deg=0.0)

        if angles_l is not None:
            limited_angles_l = []
            for j in range(3):
                # ① 60 deg/sec の速度制限を現在の位置（prev）基準で厳密にチェック
                diff = float(angles_l[j]) - float(prev_angles_l[j])
                diff_limited = np.clip(diff, -MAX_CHANGE_PER_FRAME, MAX_CHANGE_PER_FRAME)
                next_ang = prev_angles_l[j] + diff_limited

                # ② 物理的な可動限界の壁でクリップ
                next_ang_clamped = np.clip(next_ang, LIMITS_MIN[j], LIMITS_MAX[j])
                limited_angles_l.append(next_ang_clamped)

            # 【重要】制限して確定した安全な角度から、FKで「実際の現在地」を逆算
            _, _, _, p3_l = forward_kinematics_3link(*limited_angles_l)

            # 手先位置の配列を「アームが物理的に追いついた本物の座標」で上書き
            # これにより、次のフレーム（i+1）のIKは、この無理のない位置から地続きでスタートします
            y_left[i] = float(p3_l[0]) / 100.0  # タプルのX(2D上のY)
            z_left[i] = float(p3_l[1]) / 100.0  # タプルのY(2D上のZ)

            # 確定角度を保存
            prev_angles_l = limited_angles_l
        else:
            # 万が一IKの解が外れた場合は、その場に留まる（速度0）
            y_left[i] = y_left[i - 1]
            z_left[i] = z_left[i - 1]

        # ---------------------------------------------------
        # 【右腕】FKフィードバック追従ループ
        # ---------------------------------------------------
        angles_r = inverse_kinematics_3link(float(y_right[i]), float(z_right[i]), phi_deg=0.0)

        if angles_r is not None:
            limited_angles_r = []
            for j in range(3):
                # ① 速度制限
                diff = float(angles_r[j]) - float(prev_angles_r[j])
                diff_limited = np.clip(diff, -MAX_CHANGE_PER_FRAME, MAX_CHANGE_PER_FRAME)
                next_ang = prev_angles_r[j] + diff_limited

                # ② 可動限界制限
                next_ang_clamped = np.clip(next_ang, LIMITS_MIN[j], LIMITS_MAX[j])
                limited_angles_r.append(next_ang_clamped)

            # FKで実際の現在地を逆算
            _, _, _, p3_r = forward_kinematics_3link(*limited_angles_r)

            y_right[i] = float(p3_r[0]) / 100.0
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
            angles_l = inverse_kinematics_3link(float(y_l[i]), float(z_l[i]), phi_deg=0.0)
            if angles_l is not None:
                th_a_l, th_b_l, th_c_l = [np.clip(float(angles_l[j]), LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
            else:
                th_a_l, th_b_l, th_c_l = 0.0, 0.0, 0.0

            angles_r = inverse_kinematics_3link(float(y_r[i]), float(z_r[i]), phi_deg=0.0)
            if angles_r is not None:
                th_a_r, th_b_r, th_c_r = [np.clip(float(angles_r[j]), LIMITS_MIN[j], LIMITS_MAX[j]) for j in range(3)]
            else:
                th_a_r, th_b_r, th_c_r = 0.0, 0.0, 0.0

            writer.writerow([
                f"{t_steps[i]:.3f}",
                f"{th_a_l:.2f}", f"{th_b_l:.2f}", f"{th_c_l:.2f}",
                f"{th_a_r:.2f}", f"{th_b_r:.2f}", f"{th_c_r:.2f}"
            ])

    with open("delay.txt", mode='w', encoding='utf-8') as f:
        f.write(f"{time_per_row}\n")

    with open("duration.txt", mode='w', encoding='utf-8') as f:
        f.write(f"{total_duration:.2f}\n")
    print("--- 角度・設定ファイルのエクスポート完了 ---")


if __name__ == '__main__':
    midi_file = "airsulg.mid"
    export_all_files(midi_file, sampling_rate=10)