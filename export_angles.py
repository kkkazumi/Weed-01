import csv
import numpy as np
from trajectory_generator3 import generate_trajectory
from kinematics import inverse_kinematics_3link


def export_ik_angles_to_csv(midi_path, csv_output_path="angles.csv", sampling_rate=50):
    """
    MIDIファイルから全編の軌道を生成し、逆運動学(IK)で計算した
    左右アームの全関節角度の時系列データをCSVファイルに保存する
    """
    # 1. 軌道（位置情報）の一括生成
    t_steps, y_l, z_l, y_r, z_r, _ = generate_trajectory(midi_path, num_bars=None, sampling_rate=sampling_rate)

    print(f"--- 軌道データを生成しました (合計 {len(t_steps)} フレーム) ---")
    print(f"IK計算を開始し、{csv_output_path} に保存します...")

    # 2. CSVファイルの書き込み準備
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # ヘッダー（列名）の書き込み
        writer.writerow([
            "Time [s]",
            "Left_Theta_A [deg]", "Left_Theta_B [deg]", "Left_Theta_C [deg]",
            "Right_Theta_A [deg]", "Right_Theta_B [deg]", "Right_Theta_C [deg]"
        ])

        # 3. 1フレームずつIKを解いてCSVに記録
        for i in range(len(t_steps)):
            t = t_steps[i]

            # 左腕のIK計算 (届かない場合は 0.0 で代替え)
            angles_l = inverse_kinematics_3link(y_l[i], z_l[i], phi_deg=0.0)
            if angles_l is not None:
                th_a_l, th_b_l, th_c_l = angles_l
            else:
                th_a_l, th_b_l, th_c_l = 0.0, 0.0, 0.0

            # 右腕のIK計算
            angles_r = inverse_kinematics_3link(y_r[i], z_r[i], phi_deg=0.0)
            if angles_r is not None:
                th_a_r, th_b_r, th_c_r = angles_r
            else:
                th_a_r, th_b_r, th_c_r = 0.0, 0.0, 0.0

            # CSVに行を追加
            writer.writerow([
                f"{t:.3f}",
                f"{th_a_l:.2f}", f"{th_b_l:.2f}", f"{th_c_l:.2f}",
                f"{th_a_r:.2f}", f"{th_b_r:.2f}", f"{th_c_r:.2f}"
            ])

    print(f"成功！ ファイルが保存されました: {csv_output_path}")


if __name__ == '__main__':
    # あなたの環境のMIDIファイルを指定
    midi_file = "radetzky.mid"
    export_ik_angles_to_csv(midi_file)
