import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame

# ==========================================
# 1. 設定ファイルと角度CSVの読み込み
# ==========================================
csv_filename = "angles.csv"
delay_filename = "delay.txt"

if os.path.exists(delay_filename):
    with open(delay_filename, "r", encoding="utf-8") as f:
        t_interval = float(f.read().strip())
else:
    t_interval = 0.1  # デフォルト（10Hz）

if os.path.exists(csv_filename):
    print(f"[{csv_filename}] から角度データを読み込みます。")
    data = np.loadtxt(csv_filename, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    # データの各列を確実に1次元配列として抽出
    time_steps = data[:, 0]
    left_th_a = data[:, 1]
    left_th_b = data[:, 2]
    left_th_c = data[:, 3]
    right_th_a = data[:, 4]
    right_th_b = data[:, 5]
    right_th_c = data[:, 6]
    frames = len(data)
else:
    print(f"[{csv_filename}] が見つからないため、デモ用の往復データを作成します。")
    frames = 120
    time_steps = np.linspace(0, frames * t_interval, frames)
    t = np.linspace(0, 2 * np.pi, frames)
    left_th_a = -30 - 30 * np.cos(t)
    left_th_b = 120 * np.cos(t)
    left_th_c = -45 - 90 * np.cos(t)
    right_th_a = -30 - 30 * np.sin(t)
    right_th_b = 120 * np.sin(t)
    right_th_c = -45 - 90 * np.sin(t)

# ロボットのリンク長設定 (cm)
L_A = 10.0
L_B = 7.0
L_C = 10.0


# ==========================================
# 2. 順運動学(FK)の幾何学分解計算関数
# ==========================================
def calculate_fk_positions(th_a, th_b, th_c):
    """
    配列の形を崩さないよう、純粋な変数に分解して順運動学の関節位置を計算する
    """
    rad_a = np.radians(th_a)
    rad_b = np.radians(th_b)
    rad_c = np.radians(th_c)

    # 第1関節 (Arm a)
    p1_x = -L_A * np.cos(rad_a)
    p1_y = -L_A * np.sin(rad_a)
    p1_z = 0.0

    # 第2関節 (Arm b)
    axis_b_x = np.sin(rad_a)
    axis_b_y = -np.cos(rad_a)
    axis_b_z = 0.0

    p2_x = p1_x + L_B * axis_b_x
    p2_y = p1_y + L_B * axis_b_y
    p2_z = p1_z + L_B * axis_b_z

    # 第3関節・手先 (Arm c)
    cos_a, sin_a = np.cos(rad_a), np.sin(rad_a)
    cos_b, sin_b = np.cos(rad_b), np.sin(rad_b)
    dir_c_base = np.array([-cos_a * cos_b, -sin_a * cos_b, sin_b])
    norm_c = np.linalg.norm(dir_c_base)
    dir_c = dir_c_base / (norm_c if norm_c > 1e-6 else 1.0)

    axis_c = np.array([-cos_a * sin_b, -sin_a * sin_b, -cos_b])
    dir_c_rotated = dir_c * np.cos(rad_c) + np.cross(axis_c, dir_c) * np.sin(rad_c)

    p3_x = p2_x + L_C * dir_c_rotated[0]
    p3_y = p2_y + L_C * dir_c_rotated[1]
    p3_z = p2_z + L_C * dir_c_rotated[2]

    return (0.0, 0.0, 0.0), (p1_x, p1_y, p1_z), (p2_x, p2_y, p2_z), (p3_x, p3_y, p3_z)


# ==========================================
# 3. アニメーション設定と同期描画ループ
# ==========================================
def start_synchronized_viewer(midi_path):
    # Pygameで音楽再生準備
    pygame.mixer.init()
    pygame.mixer.music.load(midi_path)

    fig = plt.figure(figsize=(12, 10))
    ax_xy = fig.add_subplot(221)  # ① 左上: XY 平面図
    ax_3d = fig.add_subplot(222, projection='3d')  # ② 右上: 3D 座標
    ax_zy = fig.add_subplot(223)  # ③ 左下: ZY 平面図
    ax_xz = fig.add_subplot(224)  # ④ 右下: XZ 平面図

    fig.is_running = True
    fig.canvas.mpl_connect('close_event', lambda ev: setattr(fig, 'is_running', False))

    plt.ion()
    plt.show()

    # 過去の手先移動軌跡を蓄積するリスト
    hist_lx, hist_ly, hist_lz = [], [], []
    hist_rx, hist_ry, hist_rz = [], [], []

    # 【超重要】音楽を再生する「前に」一度ダミー描画をしてウィンドウを完全に立ち上げる（ラグ防止）
    fig.canvas.flush_events()

    # 音楽再生スタート
    pygame.mixer.music.play()

    # インデックスを追うための変数
    current_idx = 0

    # 【大改造】音楽の実際の再生位置（実時間）に完全に追従するループ
    while pygame.mixer.music.get_busy() and current_idx < frames:
        if not getattr(fig, 'is_running', True):
            break

        # Pygameから「今、音楽が何秒目か」を厳密に取得（1ミリ秒単位）
        music_time = pygame.mixer.music.get_pos() / 1000.0

        # 音楽の時間に最も近いCSVデータの行（インデックス）を探し出す
        # ※ これにより、パソコンの処理落ちや起動ラグがあっても、音と映像が絶対にズレなくなります
        current_idx = np.searchsorted(time_steps, music_time)
        if current_idx >= frames:
            break

        expected_time = time_steps[current_idx]

        # 画面のクリア
        ax_xy.cla()
        ax_3d.cla()
        ax_zy.cla()
        ax_xz.cla()

        # --- 左右アームの順運動学(FK)計算 ---
        p0_l, p1_l, p2_l, p3_l = calculate_fk_positions(left_th_a[current_idx], left_th_b[current_idx],
                                                        left_th_c[current_idx])
        p0_r, p1_r, p2_r, p3_r = calculate_fk_positions(right_th_a[current_idx], right_th_b[current_idx],
                                                        right_th_c[current_idx])

        # 【安全対策】タプルから X, Y, Z の数値を完全にバラバラに分解します
        p0_lx, p0_ly, p0_lz = p0_l;
        p1_lx, p1_ly, p1_lz = p1_l;
        p2_lx, p2_ly, p2_lz = p2_l;
        p3_lx, p3_ly, p3_lz = p3_l
        p0_rx, p0_ry, p0_rz = p0_r;
        p1_rx, p1_ry, p1_rz = p1_r;
        p2_rx, p2_ry, p2_rz = p2_r;
        p3_rx, p3_ry, p3_rz = p3_r

        # 左右のベース位置オフセット調整
        y_offset_l = -15.0
        y_offset_r = 15.0

        # 軌跡ログの保存 (Y座標にだけオフセットを加算)
        hist_lx.append(p3_lx);
        hist_ly.append(p3_ly + y_offset_l);
        hist_lz.append(p3_lz)
        hist_rx.append(p3_rx);
        hist_ry.append(p3_ry + y_offset_r);
        hist_rz.append(p3_rz)

        info_text = (
            f"Left Arm:\n  $\\theta_a$={left_th_a[current_idx]:.1f}$^\\circ$\n  $\\theta_b$={left_th_b[current_idx]:.1f}$^\\circ$\n"
            f"Right Arm:\n  $\\theta_a$={right_th_a[current_idx]:.1f}$^\\circ$\n  $\\theta_b$={right_th_b[current_idx]:.1f}$^\\circ$\n"
            f"-------\nHz: {1.0 / t_interval:.1f}")

        # ==========================================
        # ② 右上：3D ビュー (すべて数値の変数でプロット)
        # ==========================================
        # 左腕 (赤系統)
        ax_3d.plot([p0_lx, p1_lx], [p0_ly + y_offset_l, p1_ly + y_offset_l], [p0_lz, p1_lz], color='darkred',
                   linewidth=3)
        ax_3d.plot([p1_lx, p2_lx], [p1_ly + y_offset_l, p2_ly + y_offset_l], [p1_lz, p2_lz], color='red', linewidth=3)
        ax_3d.plot([p2_lx, p3_lx], [p2_ly + y_offset_l, p3_ly + y_offset_l], [p2_lz, p3_lz], color='orange',
                   linewidth=3)
        ax_3d.scatter(p3_lx, p3_ly + y_offset_l, p3_lz, color='red', s=60, edgecolors='black')
        ax_3d.plot(hist_lx, hist_ly, hist_lz, color='red', alpha=0.2, linewidth=1)

        # 右腕 (青系統)
        ax_3d.plot([p0_rx, p1_rx], [p0_ry + y_offset_r, p1_ry + y_offset_r], [p0_rz, p1_rz], color='darkblue',
                   linewidth=3)
        ax_3d.plot([p1_rx, p2_rx], [p1_ry + y_offset_r, p2_ry + y_offset_r], [p1_rz, p2_rz], color='blue', linewidth=3)
        ax_3d.plot([p2_rx, p3_rx], [p2_ry + y_offset_r, p3_ry + y_offset_r], [p2_rz, p3_rz], color='purple',
                   linewidth=3)
        ax_3d.scatter(p3_rx, p3_ry + y_offset_r, p3_rz, color='blue', s=60, edgecolors='black')
        ax_3d.plot(hist_rx, hist_ry, hist_rz, color='blue', alpha=0.2, linewidth=1)

        ax_3d.set_xlim([-25, 25]);
        ax_3d.set_ylim([-35, 35]);
        ax_3d.set_zlim([-25, 25])
        ax_3d.set_xlabel('X');
        ax_3d.set_ylabel('Y');
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D Multi-Arm View', pad=10)
        ax_3d.view_init(elev=22, azim=-60)
        ax_3d.text2D(0.02, 0.98, info_text, transform=ax_3d.transAxes, fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # ==========================================
        # ① 左上：XY 平面ビュー
        # ==========================================
        ax_xy.plot([p0_lx, p1_lx, p2_lx, p3_lx],
                   [p0_ly + y_offset_l, p1_ly + y_offset_l, p2_ly + y_offset_l, p3_ly + y_offset_l], 'r-o', linewidth=2)
        ax_xy.plot(hist_lx, hist_ly, color='red', alpha=0.2)
        ax_xy.plot([p0_rx, p1_rx, p2_rx, p3_rx],
                   [p0_ry + y_offset_r, p1_ry + y_offset_r, p2_ry + y_offset_r, p3_ry + y_offset_r], 'b-o', linewidth=2)
        ax_xy.plot(hist_rx, hist_ry, color='blue', alpha=0.2)
        ax_xy.set_xlim([-25, 25]);
        ax_xy.set_ylim([-35, 35])
        ax_xy.set_xlabel('X axis');
        ax_xy.set_ylabel('Y axis')
        ax_xy.set_title('XY Plane View (Top View)', pad=10)
        ax_xy.grid(True);
        ax_xy.set_aspect('equal')

        # ==========================================
        # ③ 左下：ZY 平面ビュー
        # ==========================================
        ax_zy.plot([p0_lz, p1_lz, p2_lz, p3_lz],
                   [p0_ly + y_offset_l, p1_ly + y_offset_l, p2_ly + y_offset_l, p3_ly + y_offset_l], 'r-o', linewidth=2)
        ax_zy.plot(hist_lz, hist_ly, color='red', alpha=0.2)
        ax_zy.plot([p0_rz, p1_rz, p2_rz, p3_rz],
                   [p0_ry + y_offset_r, p1_ry + y_offset_r, p2_ry + y_offset_r, p3_ry + y_offset_r], 'b-o', linewidth=2)
        ax_zy.plot(hist_rz, hist_ry, color='blue', alpha=0.2)
        ax_zy.set_xlim([-25, 25]);
        ax_zy.set_ylim([-35, 35])
        ax_zy.set_xlabel('Z axis');
        ax_zy.set_ylabel('Y axis')
        ax_zy.set_title('ZY Plane View (Front View)', pad=10)
        ax_zy.grid(True);
        ax_zy.set_aspect('equal')

        # ==========================================
        # ④ 右下：XZ 平面ビュー
        # ==========================================
        ax_xz.plot([p0_lx, p1_lx, p2_lx, p3_lx], [p0_lz, p1_lz, p2_lz, p3_lz], 'r-o', linewidth=2)
        ax_xz.plot(hist_lx, hist_lz, color='red', alpha=0.2)
        ax_xz.plot([p0_rx, p1_rx, p2_rx, p3_rx], [p0_rz, p1_rz, p2_rz, p3_rz], 'b-o', linewidth=2)
        ax_xz.plot(hist_rx, hist_rz, color='blue', alpha=0.2)
        ax_xz.set_xlim([-25, 25]);
        ax_xz.set_ylim([-25, 25])
        ax_xz.set_xlabel('X axis');
        ax_xz.set_ylabel('Z axis')
        ax_xz.set_title('XZ Plane View (Side View)', pad=10)
        ax_xz.grid(True);
        ax_xz.set_aspect('equal')

        # タイトルに実際の音楽の再生時間を表示
        fig.suptitle(f'Robot Arm Synchronized Drive [Music Time: {music_time:.2f}s]', fontsize=13, y=0.96)

        # 短いウェイトを入れて画面更新（CPU負荷軽減）
        plt.pause(0.01)

    pygame.mixer.music.stop()
    plt.ioff()
    print("--- 演奏およびシミュレーションが同時に終了しました ---")
    plt.show()


if __name__ == '__main__':
    midi_file = "airsulg.mid"
    start_synchronized_viewer(midi_file)
