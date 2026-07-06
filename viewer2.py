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
    t_interval = 0.1

if os.path.exists(csv_filename):
    print(f"[{csv_filename}] から角度データを読み込みます。")
    data = np.loadtxt(csv_filename, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    time_steps = data[:, 0]
    left_th_a = data[:, 1]
    left_th_b = data[:, 2]
    left_th_c = data[:, 3]
    right_th_a = data[:, 4]
    right_th_b = data[:, 5]
    right_th_c = data[:, 6]
    frames = len(data)
    print(f"データ読み込み成功: {frames} フレームの動きを再生します。")
else:
    print(f"角度データ [{csv_filename}] が存在しません。先にtrajectory_generator3.pyを実行してください。")
    exit()

# ロボットアームのリンク長 (cm)
L_A = 10.0
L_B = 7.0
L_C = 10.0


def calculate_fk_positions(th_a, th_b, th_c):
    """
    実機の角度定義に基づき、各関節の3次元(X, Y, Z)の相対座標を累積計算する
    """
    rad_a = np.radians(float(th_a))
    rad_ab = np.radians(float(th_a + th_b))
    rad_abc = np.radians(float(th_a + th_b + th_c))

    # 3軸平面アームとしての基本形 (奥行きX、高さY)
    p0_x, p0_y, p0_z = 0.0, 0.0, 0.0
    p1_x = L_A * np.cos(rad_a)
    p1_y = L_A * np.sin(rad_a)

    p2_x = p1_x + L_B * np.cos(rad_ab)
    p2_y = p1_y + L_B * np.sin(rad_ab)

    p3_x = p2_x + L_C * np.cos(rad_abc)
    p3_y = p2_y + L_C * np.sin(rad_abc)

    return (p0_x, p0_y, p0_z), (p1_x, p1_y, 0.0), (p2_x, p2_y, 0.0), (p3_x, p3_y, 0.0)


# ==========================================
# 2. 同期描画システム (4画面マルチアングル版)
# ==========================================
def start_synchronized_viewer(midi_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(midi_path)
        has_music = True
    except Exception as e:
        print(f"音楽再生をスキップします: {e}")
        has_music = False

    # 豪華な2行×2列の4画面マルチモニターを初期化
    fig = plt.figure(figsize=(12, 10))
    ax_xy = fig.add_subplot(221)  # ① 左上: XY平面（真上）
    ax_3d = fig.add_subplot(222, projection='3d')  # ② 右上: 3Dマルチ
    ax_zy = fig.add_subplot(223)  # ③ 左下: ZY平面（正面）
    ax_xz = fig.add_subplot(224)  # ④ 右下: XZ平面（真横）

    # 2Dグラフのレイアウト固定設定
    for ax in [ax_xy, ax_zy, ax_xz]:
        ax.set_xlim([-25, 25]);
        ax.set_ylim([-35, 35]);
        ax.grid(True);
        ax.set_aspect('equal')

    ax_xy.set_title("XY Plane View (Top View)");
    ax_xy.set_xlabel("X (Depth)");
    ax_xy.set_ylabel("Y (Left-Right)")
    ax_zy.set_title("ZY Plane View (Front View)");
    ax_zy.set_xlabel("Z (Up-Down)");
    ax_zy.set_ylabel("Y (Left-Right)")
    ax_xz.set_title("XZ Plane View (Side View)");
    ax_xz.set_xlabel("X (Depth)");
    ax_xz.set_ylabel("Z (Up-Down)")

    # 3Dグラフのレイアウト設定
    ax_3d.set_xlim([-25, 25]);
    ax_3d.set_ylim([-35, 35]);
    ax_3d.set_zlim([-25, 25])
    ax_3d.set_xlabel('X (Depth)');
    ax_3d.set_ylabel('Y (Left-Right)');
    ax_3d.set_zlabel('Z (Up-Down)')
    ax_3d.set_title('3D Multi-Arm View', pad=10)
    ax_3d.view_init(elev=22, azim=-60)

    # 描画オブジェクトの先行生成 (オブジェクトを再利用してフリーズを完璧に防止)
    line_xy_l, = ax_xy.plot([], [], 'r-o', linewidth=3);
    line_xy_r, = ax_xy.plot([], [], 'b-o', linewidth=3)
    hist_xy_l, = ax_xy.plot([], [], 'r-', alpha=0.15);
    hist_xy_r, = ax_xy.plot([], [], 'b-', alpha=0.15)

    line_zy_l, = ax_zy.plot([], [], 'r-o', linewidth=3);
    line_zy_r, = ax_zy.plot([], [], 'b-o', linewidth=3)
    hist_zy_l, = ax_zy.plot([], [], 'r-', alpha=0.15);
    hist_zy_r, = ax_zy.plot([], [], 'b-', alpha=0.15)

    line_xz_l, = ax_xz.plot([], [], 'r-o', linewidth=3);
    line_xz_r, = ax_xz.plot([], [], 'b-o', linewidth=3)
    hist_xz_l, = ax_xz.plot([], [], 'r-', alpha=0.15);
    hist_xz_r, = ax_xz.plot([], [], 'b-', alpha=0.15)

    line_3d_l, = ax_3d.plot([], [], [], 'r-o', linewidth=3);
    line_3d_r, = ax_3d.plot([], [], [], 'b-o', linewidth=3)
    track_3d_l, = ax_3d.plot([], [], [], 'r-', alpha=0.15);
    track_3d_r, = ax_3d.plot([], [], [], 'b-', alpha=0.15)

    fig.is_running = True
    fig.canvas.mpl_connect('close_event', lambda ev: setattr(fig, 'is_running', False))

    plt.ion()
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # 過去の移動軌跡を蓄積するリスト
    hist_lx, hist_ly, hist_lz = [], [], []
    hist_rx, hist_ry, hist_rz = [], [], []

    if has_music:
        pygame.mixer.music.play()

    while has_music and not pygame.mixer.music.get_busy():
        time.sleep(0.01)

    start_real_time = time.time()

    while True:
        if not getattr(fig, 'is_running', True):
            break

        raw_pos = pygame.mixer.music.get_pos()
        if raw_pos < 0 and not pygame.mixer.music.get_busy():
            break

        music_time = max(0.0, raw_pos / 1000.0)
        current_idx = int(music_time / t_interval)

        if current_idx >= frames:
            break

        expected_time = time_steps[current_idx]

        if music_time < expected_time:
            time.sleep(expected_time - music_time)

        # リアルタイム角度をコンソール（画面下部）に出力
        print(
            f"[Time: {expected_time:5.2f}s] "
            f"LEFT: A={left_th_a[current_idx]:6.1f}, B={left_th_b[current_idx]:6.1f}, C={left_th_c[current_idx]:6.1f} | "
            f"RIGHT: A={right_th_a[current_idx]:6.1f}, B={right_th_b[current_idx]:6.1f}, C={right_th_c[current_idx]:6.1f}"
        )

        # 順運動学(FK)計算
        p0_l, p1_l, p2_l, p3_l = calculate_fk_positions(left_th_a[current_idx], left_th_b[current_idx],
                                                        left_th_c[current_idx])
        p0_r, p1_r, p2_r, p3_r = calculate_fk_positions(right_th_a[current_idx], right_th_b[current_idx],
                                                        right_th_c[current_idx])

        p0_lx, p0_ly, p0_lz = p0_l;
        p1_lx, p1_ly, p1_lz = p1_l;
        p2_lx, p2_ly, p2_lz = p2_l;
        p3_lx, p3_ly, p3_lz = p3_l
        p0_rx, p0_ry, p0_rz = p0_r;
        p1_rx, p1_ry, p1_rz = p1_r;
        p2_rx, p2_ry, p2_rz = p2_r;
        p3_rx, p3_ry, p3_rz = p3_r

        # 左右のベース位置オフセット設定 (左はY=-15cm, 右はY=+15cm)
        y_offset_l = -15.0
        y_offset_r = 15.0

        # 手先位置の軌跡データを蓄積 (YZ正面対称に対応)
        hist_lx.append(p3_lx);
        hist_ly.append(p3_ly + y_offset_l);
        hist_lz.append(p3_lz)
        hist_rx.append(p3_rx);
        hist_ry.append(p3_ry + y_offset_r);
        hist_rz.append(p3_rz)

        # 各プロットの高速上書き更新
        # ① XY平面（真上から）
        line_xy_l.set_data([p0_lx, p1_lx, p2_lx, p3_lx],
                           [p0_ly + y_offset_l, p1_ly + y_offset_l, p2_ly + y_offset_l, p3_ly + y_offset_l])
        line_xy_r.set_data([p0_rx, p1_rx, p2_rx, p3_rx],
                           [p0_ry + y_offset_r, p1_ry + y_offset_r, p2_ry + y_offset_r, p3_ry + y_offset_r])
        hist_xy_l.set_data(hist_lx, hist_ly);
        hist_xy_r.set_data(hist_rx, hist_ry)

        # ② ZY平面（正面から）
        line_zy_l.set_data([p0_lz, p1_lz, p2_lz, p3_lz],
                           [p0_ly + y_offset_l, p1_ly + y_offset_l, p2_ly + y_offset_l, p3_ly + y_offset_l])
        line_zy_r.set_data([p0_rz, p1_rz, p2_rz, p3_rz],
                           [p0_ry + y_offset_r, p1_ry + y_offset_r, p2_ry + y_offset_r, p3_ry + y_offset_r])
        hist_zy_l.set_data(hist_lz, hist_ly);
        hist_zy_r.set_data(hist_rz, hist_ry)

        # ③ XZ平面（真横から）
        line_xz_l.set_data([p0_lx, p1_lx, p2_lx, p3_lx], [p0_lz, p1_lz, p2_lz, p3_lz])
        line_xz_r.set_data([p0_rx, p1_rx, p2_rx, p3_rx], [p0_rz, p1_rz, p2_rz, p3_rz])
        hist_xz_l.set_data(hist_lx, hist_lz);
        hist_xz_r.set_data(hist_rx, hist_rz)

        # ④ 3Dマルチアームビュー 【タイポ行を削除し、純粋な1次元配列としてZ軸を確実に更新します】
        line_3d_l.set_data([p0_lx, p1_lx, p2_lx, p3_lx],
                           [p0_ly + y_offset_l, p1_ly + y_offset_l, p2_ly + y_offset_l, p3_ly + y_offset_l])
        line_3d_l.set_3d_properties(np.array([p0_lz, p1_lz, p2_lz, p3_lz], dtype=float))

        line_3d_r.set_data([p0_rx, p1_rx, p2_rx, p3_rx],
                           [p0_ry + y_offset_r, p1_ry + y_offset_r, p2_ry + y_offset_r, p3_ry + y_offset_r])
        line_3d_r.set_3d_properties(np.array([p0_rz, p1_rz, p2_rz, p3_rz], dtype=float))

        track_3d_l.set_data(hist_lx, hist_ly);
        track_3d_l.set_3d_properties(np.array(hist_lz, dtype=float))
        track_3d_r.set_data(hist_rx, hist_ry);
        track_3d_r.set_3d_properties(np.array(hist_rz, dtype=float))

        # 全体タイトルの更新
        fig.suptitle(f'Robot Arm Synchronized Drive [Music Time: {music_time:.2f}s]', fontsize=13)

        # 画面の即時リフレッシュ強制執行
        fig.canvas.draw()
        fig.canvas.flush_events()

    if has_music:
        pygame.mixer.music.stop()
    plt.ioff()
    print("--- シミュレーション正常終了 ---")
    plt.show()


if __name__ == '__main__':
    midi_file = "radetzky.mid"
    start_synchronized_viewer(midi_file)
