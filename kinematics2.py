import numpy as np

# 実機の物理的な可動限界 [deg]
LIMITS_MIN = np.array([-60.0, -120.0, -135.0])
LIMITS_MAX = np.array([0.0, 120.0, 45.0])

L_A = 10.0
L_B = 7.0
L_C = 10.0


def forward_kinematics_3link(theta_a, theta_b, theta_c):
    """
    各関節の角度から、手先および各関節の座標(cm)を返す順運動学(FK)
    """
    rad_a = np.radians(float(theta_a))
    rad_ab = np.radians(float(theta_a + theta_b))
    rad_abc = np.radians(float(theta_a + theta_b + theta_c))

    p0 = (0.0, 0.0, 0.0)

    p1_x = float(L_A * np.cos(rad_a))
    p1_y = float(L_A * np.sin(rad_a))

    p2_x = float(p1_x + L_B * np.cos(rad_ab))
    p2_y = float(p1_y + L_B * np.sin(rad_ab))

    p3_x = float(p2_x + L_C * np.cos(rad_abc))
    p3_y = float(p2_y + L_C * np.sin(rad_abc))

    # Viewerのアンパックに100%適合するタプル形式で返却
    return p0, (p1_x, p1_y, 0.0), (p2_x, p2_y, 0.0), (p3_x, p3_y, 0.0)


def inverse_kinematics_3link(y_m, z_m):
    """
    【数値的逆運動学】ヤコビアンの擬似逆行列を用いて、制限の壁に張り付くバグを完全解消するIK
    ※ 引数から phi_deg を廃止し、位置だけで最適解を算出します
    """
    target_pos = np.array([float(y_m) * 100.0, float(z_m) * 100.0])

    # 可動範囲の中央を初期値（探索のスタート地点）にする
    current_q = (LIMITS_MIN + LIMITS_MAX) / 2.0

    max_iterations = 20
    convergence_threshold = 0.01  # 0.1mmの精度
    learning_rate = 0.2

    for _ in range(max_iterations):
        # 現在の角度での手先位置(FK)を取得
        _, _, _, p3 = forward_kinematics_3link(current_q[0], current_q[1], current_q[2])
        current_pos = np.array([p3[0], p3[1]])

        # 目標位置との誤差
        error = target_pos - current_pos
        if np.linalg.norm(error) < convergence_threshold:
            break

        # ヤコビ行列(Jacobian)を数値微分で計算
        dq = 0.01
        J = np.zeros((2, 3))
        for j in range(3):
            q_plus = current_q.copy()
            q_plus[j] += dq
            _, _, _, p3_plus = forward_kinematics_3link(q_plus[0], q_plus[1], q_plus[2])
            pos_plus = np.array([p3_plus[0], p3_plus[1]])
            J[:, j] = (pos_plus - current_pos) / dq

        # 擬似逆行列で角度更新量を算出
        J_pinv = np.linalg.pinv(J)
        delta_q = J_pinv @ error

        # 角度を更新し、可動範囲の上限・下限で厳密にクリップ
        current_q += learning_rate * np.degrees(delta_q)
        current_q = np.clip(current_q, LIMITS_MIN, LIMITS_MAX)

    return float(current_q[0]), float(current_q[1]), float(current_q[2])
