import numpy as np

# 実機の物理的な可動限界 [deg]
LIMITS_MIN = np.array([-60.0, -120.0, -135.0])
LIMITS_MAX = np.array([0.0, 120.0, 45.0])

L_A = 10.0
L_B = 7.0
L_C = 10.0

# 前回の計算成功角度を保持するグローバル変数（解の跳躍を防止する基準点）
_prev_angles_left = np.array([-30.0, -60.0, 35.0])
_prev_angles_right = np.array([-30.0, -60.0, 35.0])


def forward_kinematics_3link(theta_a, theta_b, theta_c):
    """各関節の角度から、手先および各関節の座標(cm)を返す順運動学(FK)"""
    rad_a = np.radians(float(theta_a))
    rad_ab = np.radians(float(theta_a + theta_b))
    rad_abc = np.radians(float(theta_a + theta_b + theta_c))

    p0 = (0.0, 0.0, 0.0)
    p1_y = float(L_A * np.cos(rad_a))
    p1_z = float(L_A * np.sin(rad_a))
    p2_y = float(p1_y + L_B * np.cos(rad_ab))
    p2_z = float(p1_z + L_B * np.sin(rad_ab))
    p3_y = float(p2_y + L_C * np.cos(rad_abc))
    p3_z = float(p2_z + L_C * np.sin(rad_abc))

    return p0, (p1_y, p1_z, 0.0), (p2_y, p2_z, 0.0), (p3_y, p3_z, 0.0)


def inverse_kinematics_3link(y_m, z_m):
    """
    【解の跳躍防止型・逆運動学】
    幾何学的な2つの折り畳み候補（Elbow-Up / Down）を両方算出し、
    前回の姿勢から最も移動量が少ない（地続きで滑らかな）候補を厳密に自動選択します。
    """
    global _prev_angles_left, _prev_angles_right

    y = float(y_m) * 100.0
    z = float(z_m) * 100.0

    # 左右どちらのアームの計算かを目標座標の正負（左は負、右は正）で自動判定
    is_right = (y_m > 0)
    prev_q = _prev_angles_right if is_right else _prev_angles_left

    phi_rad = np.arctan2(z, y)
    y_w = y - L_C * np.cos(phi_rad)
    z_w = z - L_C * np.sin(phi_rad)

    cos_theta_b = (y_w ** 2 + z_w ** 2 - L_A ** 2 - L_B ** 2) / (2.0 * L_A * L_B)
    cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)

    # 候補①：Elbow-Down解（肘を下側に折る姿勢）
    sin_b1 = -np.sqrt(1.0 - cos_theta_b ** 2)
    th_b1_rad = np.arctan2(sin_b1, cos_theta_b)
    th_a1_rad = np.arctan2(z_w, y_w) - np.arctan2(L_B * sin_b1, L_A + L_B * cos_theta_b)
    th_a1_rad = np.arctan2(np.sin(th_a1_rad), np.cos(th_a1_rad))
    th_c1_rad = phi_rad - th_a1_rad - th_b1_rad

    q1 = np.array([np.degrees(th_a1_rad), np.degrees(th_b1_rad), np.degrees(th_c1_rad)])
    q1 = np.clip(q1, LIMITS_MIN, LIMITS_MAX)

    # 候補②：Elbow-Up解（肘を上側に折る姿勢）
    sin_b2 = np.sqrt(1.0 - cos_theta_b ** 2)
    th_b2_rad = np.arctan2(sin_b2, cos_theta_b)
    th_a2_rad = np.arctan2(z_w, y_w) - np.arctan2(L_B * sin_b2, L_A + L_B * cos_theta_b)
    th_a2_rad = np.arctan2(np.sin(th_a2_rad), np.cos(th_a2_rad))
    th_c2_rad = phi_rad - th_a2_rad - th_b2_rad

    q2 = np.array([np.degrees(th_a2_rad), np.degrees(th_b2_rad), np.degrees(th_c2_rad)])
    q2 = np.clip(q2, LIMITS_MIN, LIMITS_MAX)

    # 前回の角度(prev_q)と比べて、角度の変化総量（距離）が少ない方を採用
    dist1 = np.linalg.norm(q1 - prev_q)
    dist2 = np.linalg.norm(q2 - prev_q)

    best_q = q1 if dist1 <= dist2 else q2

    # 次回のために最新の確定角度をグローバルに記憶
    if is_right:
        _prev_angles_right = best_q.copy()
    else:
        _prev_angles_left = best_q.copy()

    return float(best_q[0]), float(best_q[1]), float(best_q[2])
