import numpy as np


def inverse_kinematics_3link(y_m, z_m, L_A=10.0, L_B=7.0, L_C=10.0):
    """
    平面3リンクアームの逆運動学 (IK) - 実機可動範囲（下向き基準）完全適合版
    :param y_m: 手先の目標Y座標 [メートル]
    :param z_m: 手先の目標Z座標 [メートル]
    """
    # 1. 単位を [m] から [cm] へ変換
    y = float(y_m) * 100.0
    z = float(z_m) * 100.0

    # 手先（リンクC）の傾きphiは、ベースから目標点への直線の傾きに自然に追従させ、
    # 関節への無理な折り曲げ負荷を逃がします
    phi_rad = np.arctan2(z, y)

    # 2. リンクCの根元（手首位置: y_w, z_w）を逆算
    y_w = y - L_C * np.cos(phi_rad)
    z_w = z - L_C * np.sin(phi_rad)

    # 3. 余弦定理から第2関節（theta_b）の角度を計算
    cos_theta_b = (y_w ** 2 + z_w ** 2 - L_A ** 2 - L_B ** 2) / (2.0 * L_A * L_B)
    cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)

    # 実機の構造（肘を下側に折り曲げるElbow-down解）を選択
    sin_theta_b = -np.sqrt(1.0 - cos_theta_b ** 2)
    theta_b_rad = np.arctan2(sin_theta_b, cos_theta_b)

    # 4. 第1関節（theta_a）の幾何学計算
    k1 = L_A + L_B * cos_theta_b
    k2 = L_B * sin_theta_b
    theta_a_rad = np.arctan2(z_w, y_w) - np.arctan2(k2, k1)
    theta_a_rad = np.arctan2(np.sin(theta_a_rad), np.cos(theta_a_rad))

    # 第3関節（theta_c）の計算
    theta_c_rad = phi_rad - theta_a_rad - theta_b_rad

    # 度数法 [deg] に変換
    th_a = float(np.degrees(theta_a_rad))
    th_b = float(np.degrees(theta_b_rad))
    th_c = float(np.degrees(theta_c_rad))

    return th_a, th_b, th_c


def forward_kinematics_3link(theta_a, theta_b, theta_c, L_A=10.0, L_B=7.0, L_C=10.0):
    """
    各関節の座標を正確に出すための順運動学 (FK)
    """
    rad_a = np.radians(float(theta_a))
    rad_ab = np.radians(float(theta_a + theta_b))
    rad_abc = np.radians(float(theta_a + theta_b + theta_c))

    p0 = (0.0, 0.0, 0.0)

    # 各関節の位置を地続きで累積足し算
    p1_y = float(L_A * np.cos(rad_a))
    p1_z = float(L_A * np.sin(rad_a))

    p2_y = float(p1_y + L_B * np.cos(rad_ab))
    p2_z = float(p1_z + L_B * np.sin(rad_ab))

    p3_y = float(p2_y + L_C * np.cos(rad_abc))
    p3_z = float(p2_z + L_C * np.sin(rad_abc))

    # 2D描画に必要な [Y座標, Z座標] の形式に統一して返却
    return p0, (p1_y, p1_z, 0.0), (p2_y, p2_z, 0.0), (p3_y, p3_z, 0.0)
