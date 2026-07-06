import numpy as np


def inverse_kinematics_3link(y_m, z_m, phi_deg=0.0, L_A=10.0, L_B=7.0, L_C=10.0):
    """
    平面3リンクアームの逆運動学 (IK) - 実機可動範囲適合版
    """
    y = float(y_m) * 100.0
    z = float(z_m) * 100.0
    phi = np.radians(float(phi_deg))

    y_w = y - L_C * np.cos(phi)
    z_w = z - L_C * np.sin(phi)

    cos_theta_b = (y_w ** 2 + z_w ** 2 - L_A ** 2 - L_B ** 2) / (2.0 * L_A * L_B)
    cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)

    # Elbow-down 解
    sin_theta_b = -np.sqrt(1.0 - cos_theta_b ** 2)
    theta_b_rad = np.arctan2(sin_theta_b, cos_theta_b)

    k1 = L_A + L_B * cos_theta_b
    k2 = L_B * sin_theta_b
    theta_a_rad = np.arctan2(z_w, y_w) - np.arctan2(k2, k1)
    theta_a_rad = np.arctan2(np.sin(theta_a_rad), np.cos(theta_a_rad))

    theta_c_rad = phi - theta_a_rad - theta_b_rad

    return float(np.degrees(theta_a_rad)), float(np.degrees(theta_b_rad)), float(np.degrees(theta_c_rad))


def forward_kinematics_3link(theta_a, theta_b, theta_c, L_A=10.0, L_B=7.0, L_C=10.0):
    """
    各関節の座標を出すための順運動学 (FK)
    形状エラーを防ぐため、NumPy配列ではなく純粋なPythonのfloatリストで値を返します
    """
    rad_a = np.radians(float(theta_a))
    rad_ab = np.radians(float(theta_a + theta_b))
    rad_abc = np.radians(float(theta_a + theta_b + theta_c))

    # 形状エラーを完全に防ぐため、各関節座標を[Y, Z]のリストとして順に計算
    p0 = [0.0, 0.0]
    p1 = [float(L_A * np.cos(rad_a)), float(L_A * np.sin(rad_a))]
    p2 = [float(p1[0] + L_B * np.cos(rad_ab)), float(p1[1] + L_B * np.sin(rad_ab))]
    p3 = [float(p2[0] + L_C * np.cos(rad_abc)), float(p2[1] + L_C * np.sin(rad_abc))]

    return p0, p1, p2, p3
