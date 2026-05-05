import numpy as np
from math import cos, sin, pi, sqrt, atan2, acos
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy import integrate, optimize
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import copy


class manta(object):

    def __init__(self, ):
        # geometrical parameters
        self.phi = 0.  # roll
        self.theta = 0.  # pitch
        self.psi = 0.  # yaw
        self.theta1_l = 0.  # left bending angle of fin
        self.theta1_r = 0.  # right bending angle of fin
        self.d_theta1_l = 0.  # left bending velocity of fin
        self.d_theta1_r = 0.  # right bending velocity of fin
        self.beta_l = 0.  # left phase difference
        self.beta_r = 0.  # right phase difference
        self.R_wb = np.eye(3)  # rotation matrix of rigide body
        self.omega_bl = 0.  # left fin angular velocity
        self.omega_br = 0.  # right fin angular velocity
        self.theta_bias_s = 2 * pi / 9
        self.theta_bias_l = pi / 3

        self.W = 0.14  # fin width
        self.P_bl = np.array([98.79, 0.93, -10.4]) / 1000  # position vector of Ol w.r.t Cb
        self.P_br = np.array([-98.79, 0.93, -10.4]) / 1000
        self.P_lc = np.array([39.8, 74.4, 0]) / 1000  # something wrong for this data
        self.P_rc = np.array([-39.8, 74.4, 0]) / 1000
        self.P_cd = np.array([0.056, 0, 0])
        self.P_wb = np.zeros((3))
        # Kinematic parameters

        self.Vb = np.zeros((3))  # rigid body linear velocity
        self.Omega_b = np.zeros((3))  # rigid body angular velocity
        self.Cn1 = np.diag([0.5, 1.8, 0.5])  # hydrodynamic coefficient of infinitesimal element 1
        self.Cn2 = np.diag([0.5, 1.8, 0.5])  # hydrodynamic coefficient of infinitesimal element 2
        self.Cp1 = np.diag([0.15, 1, 1])  # np.diag([1, 0.15, 1])  # contribution of the velocity lV_Pi
        self.Cp2 = np.diag([0.15, 1, 1])  # np.diag([1, 0.15, 1])  # contribution of the velocity lV_Pi
        self.CR1 = 0.2
        self.CR2 = 0.3
        self.CE1 = 0.8
        self.CE2 = 0.7
        self.CB = np.diag([1, 0.2, 50]) * 0.672  # np.diag((0.05,0.025,0.05))
        self.Cw = np.diag([0.015, 0.010, 0.020])  # np.diag((0.01,0.015,0.02))
        self.A = np.diag([0.0298, 0.0167, 0.0577])
        self.J = np.diag([0.0218, 0.022, 0.0396])
        self.inv_J = np.linalg.inv(self.J)
        self.m = 3.68

    def W_R_b(self):
        '''
        cθcψ, sφsθcψ − cφsψ, cφsθcψ + sφsψ
        cθsψ,sφsθsψ + cφcψ, cφsθcψ − sφcψ
        −sθ, sφcθ, cφcθ
        '''
        ct, st = cos(self.theta), sin(self.theta)
        ch, sh = cos(self.phi), sin(self.phi)
        cp, sp = cos(self.psi), sin(self.psi)
        self.R_wb = np.array([[ct * cp, sh * st * cp - ch * sp, ch * st * cp + sh * sp],
                              [ct * sp, sh * st * sp + ch * cp, ch * st * sp - sh * cp],
                              [-st, sh * ct, ch * ct]])
        return self.R_wb

    def bending_fin(self, theta1):
        '''

        :param theta1: bending angle theta1
        :return: rotation matrix R_bl
        '''
        self.R_bl = np.array([[cos(theta1), -sin(theta1), 0],
                              [sin(theta1), cos(theta1), 0],
                              [0., 0., 1.]])
        return self.R_bl

    ######### 计算s处鳍条转动角度
    def slice_fin_angle(self, s, t, left):
        '''

        :param s:
        :param t:
        :param left:
        :return: angle
        '''
        if left == True:
            theta_b = self.omega_bl * t - s / self.W * self.beta_l
        else:
            theta_b = self.omega_br * t - s / self.W * self.beta_r

        if theta_b < 0:
            temp = theta_b / (2 * pi)
            theta_b = theta_b - int(temp - 1) * 2 * pi
        elif theta_b > 2 * pi:
            temp = theta_b / (2 * pi)
            theta_b = theta_b - int(temp) * 2 * pi
        a = 0.024
        b = 0.010
        c = 0.014
        e = 0.024
        # theta_b = 200/180*pi

        u = sqrt(a ** 2 + b ** 2 - 2 * a * b * cos(theta_b))
        theta_r3 = acos((e ** 2 + c ** 2 - u ** 2) / (2 * e * c))
        theta_r1 = acos((e ** 2 + u ** 2 - c ** 2) / (2 * e * u))
        theta_r2 = acos((a ** 2 + u ** 2 - b ** 2) / (2 * a * u))
        if theta_b <= pi:
            theta_s = self.theta_bias_s - (theta_r1 + theta_r2)
        else:
            theta_s = self.theta_bias_s - (theta_r1 - theta_r2)
        theta_l = self.theta_bias_l - theta_r3
        if left == False:
            theta_s = - theta_s
            theta_l = - theta_l
        L1 = 0.056
        k = 0.0045
        l2 = 0.168 - k * s ** 2

        # middle point of 2 rods in frame Cl
        if left == True:
            P1 = np.array([L1 * cos(theta_s) / 2, 0, - L1 * sin(theta_s) / 2])
            P2 = np.array([l2 * cos(theta_s + theta_l) / 2, 0, - l2 * sin(theta_s + theta_l) / 2])
        else:
            P1 = np.array([-L1 * cos(theta_s) / 2, 0, L1 * sin(theta_s) / 2])
            P2 = np.array([-l2 * cos(theta_s + theta_l) / 2, 0, l2 * sin(theta_s + theta_l) / 2])
        return theta_s, theta_l, P1, P2



    ### 计算鳍条曲线I1，I2上点的坐标以及局部切向量P1，P2
    def fin_curve_point(self, s, t, left):
        '''

        :param s:
        :param t:
        :param left:
        :return: coordinates of point on curve I1, I2,  middle point vectors P1,P2
        '''
        if left == True:
            theta_b = self.omega_bl * t - s / self.W * self.beta_l
        else:
            theta_b = self.omega_br * t - s / self.W * self.beta_r
        if theta_b < 0:
            temp = theta_b / (2 * pi)
            theta_b = theta_b - int(temp - 1) * 2 * pi
        elif theta_b > 2 * pi:
            temp = theta_b / (2 * pi)
            theta_b = theta_b - int(temp) * 2 * pi
        a = 0.024
        b = 0.010
        c = 0.014
        e = 0.024
        # theta_b = 200/180*pi

        u = sqrt(a ** 2 + b ** 2 - 2 * a * b * cos(theta_b))
        theta_r3 = acos((e ** 2 + c ** 2 - u ** 2) / (2 * e * c))
        theta_r1 = acos((e ** 2 + u ** 2 - c ** 2) / (2 * e * u))
        theta_r2 = acos((a ** 2 + u ** 2 - b ** 2) / (2 * a * u))
        if theta_b <= pi:
            theta_s = self.theta_bias_s - (theta_r1 + theta_r2)
        else:
            theta_s = self.theta_bias_s - (theta_r1 - theta_r2)
        theta_l = self.theta_bias_l - theta_r3
        if left == False:
            theta_s = - theta_s
            theta_l = - theta_l
        L1 = 0.056
        k = 0.0045
        l2 = 0.168 - k * s ** 2

        if left == True:
            P_lc = np.array([0.01838507, self.W / 2, -0.0154269])
            I1 = np.array([P_lc[0] + L1 * cos(theta_s), P_lc[1] - s, P_lc[2] - L1 * sin(theta_s)])
            I2 = np.array([I1[0] + l2 * cos(theta_s + theta_l), I1[1], I1[2] - l2 * sin(theta_s + theta_l)])
            P1 = np.array([P_lc[0] + L1 * cos(theta_s) / 2, I1[1], P_lc[2] - L1 * sin(theta_s) / 2])
            P2 = np.array([I1[0] + l2 * cos(theta_s + theta_l) / 2, I1[1], I1[2] - l2 * sin(theta_s + theta_l) / 2])
        else:
            P_lc = np.array([-0.01838507, self.W / 2, -0.0154269])
            I1 = np.array([P_lc[0] - L1 * cos(theta_s), P_lc[1] - s, P_lc[2] + L1 * sin(theta_s)])
            I2 = np.array([I1[0] - l2 * cos(theta_s + theta_l), I1[1], I1[2] + l2 * sin(theta_s + theta_l)])
            P1 = np.array([P_lc[0] - L1 * cos(theta_s) / 2, P_lc[1] - s, P_lc[2] + L1 * sin(theta_s) / 2])
            P2 = np.array([I1[0] - l2 * cos(theta_s + theta_l) / 2, I1[1], I1[2] + l2 * sin(theta_s + theta_l) / 2])
        return I1, I2, P1, P2


    ########### 躯干及刚性传动部分的速度传递
    def rigid_body_kinematic(self):
        self.V_wb = self.R_wb @ self.Vb
        self.Omega_wb = self.R_wb @ self.Omega_b
        #R.T基体转到fin坐标系
        # velocity of Ol of left fin in frame Cl
        R_lb = np.transpose(self.bending_fin(self.theta1_l))
        self.Vl = R_lb @ (self.Vb + np.cross(self.Omega_b, self.P_bl))
        self.Omega_l = R_lb @ self.Omega_b + self.d_theta1_l * np.array([0, 0, 1])
        # velocity of Or of right fin in frame Cr
        R_rb = np.transpose(self.bending_fin(self.theta1_r))
        self.Vr = R_rb @ (self.Vb + np.cross(self.Omega_b, self.P_br))
        self.Omega_r = R_rb @ self.Omega_b + self.d_theta1_r * np.array([0, 0, 1])

    ########### 鳍条角速度的计算（使用数值微分）
    def fin_rotation_velocity(self, t, s, left):
        theta_2_p, theta_3_p, _, _ = self.slice_fin_angle(s, t + 0.00001, left)
        theta_2_m, theta_3_m, _, _ = self.slice_fin_angle(s, t - 0.00001, left)
        return (theta_2_p - theta_2_m) / 0.00002, (theta_3_p - theta_3_m) / 0.00002


    ########### 柔性鳍条的速度传递及点P1，P2坐标的计算
    def fin_slice_kinematic(self, s, t, left):
        '''
        :param s: coordinate [0,self.W]
        :param t: time
        :param left: left or right fin?
        :return: velocities and coordinates of 2 middles points in frame Cl
        '''
        # Before use this function, don't forget updating the rigid body kinematic with self.rigid_body_kinematic

        theta_2, theta_3, P1, P2 = self.slice_fin_angle(s, t, left)
        R_cl = np.array([[cos(theta_2), 0, -sin(theta_2)], [sin(theta_2), 0, cos(theta_2)], [0, 1, 0]])
        R_lc = np.transpose(R_cl)
        P_lc = np.array([0.01838507, self.W / 2 - s, -0.0154269])  # there is a prob in paper
        if left == False:
            P_lc[0] = - P_lc[0]
        # velocity of Oc in frame Cc
        if left == True:
            Vc = R_cl @ (self.Vl + np.cross(self.Omega_l, P_lc))
            domega_2, dtheta_3 = self.fin_rotation_velocity(t, s, left)
            Omega_c = R_cl @ self.Omega_l + domega_2 * np.array([0, 1, 0])
            # velocity of middle point P1 in frame Cl
            lVc = R_lc @ Vc + np.cross(P_lc, (R_lc @ Omega_c))
            l_Omega_c = self.Omega_l + domega_2 * np.array([0, 1, 0])

        else:
            Vc = R_cl @ (self.Vr + np.cross(self.Omega_r, P_lc))
            domega_2, dtheta_3 = self.fin_rotation_velocity(t, s, left)
            Omega_c = R_cl @ self.Omega_r + domega_2 * np.array([0, 1, 0])
            # velocity of middle point P1 in frame Cl
            lVc = R_lc @ Vc + np.cross(P_lc, (R_lc @ Omega_c))
            l_Omega_c = self.Omega_r + domega_2 * np.array([0, 1, 0])

        lV_P1 = lVc + np.cross(l_Omega_c, P1)
        R_dc = np.array([[cos(theta_3), 0, -sin(theta_3)], [sin(theta_3), 0, cos(theta_3)], [0, 1, 0]])
        R_ld = R_lc @ np.transpose(R_dc)
        # velociy of Od in frame Cd
        if left == True:
            P_cd = self.P_cd
        else:
            P_cd = - self.P_cd
        Vd = R_dc @ (Vc + np.cross(Omega_c, P_cd))
        Omega_d = R_dc @ Omega_c + dtheta_3 * np.array([0, 1, 0])
        l_Omega_d = R_ld @ Omega_d
        lVd = R_ld @ Vd + (P_lc + R_lc @ P_cd)
        lV_P2 = lVd + np.cross(l_Omega_d, P2)
        return lV_P1, lV_P2, P1, P2

    ########### 计算作用在一系列微元上的莫里森水动力
    def slice_hydrodynamic_force(self, list_s, t, left):
        dW = np.zeros((6, len(list_s)))
        for i in range(len(list_s)):
            s = list_s[i]
            lV_P1, lV_P2, P1, P2 = self.fin_slice_kinematic(s, t, left)
            ############# Calculate the tangent direction
            R1 = np.array([0, -1, 0])
            _, _, Q1, Q2 = self.fin_curve_point(s, t, left)
            I1_p, I2_p, _, _ = self.fin_curve_point(s + self.W / 1000, t, left)
            I1_m, I2_m, _, _ = self.fin_curve_point(s - self.W / 1000, t, left)
            E1, E2 = (I1_p - I1_m) / (self.W / 500), (I2_p - I2_m) / (self.W / 500)
            E1 = E1 / np.linalg.norm(E1)
            E2 = E2 / np.linalg.norm(E2)

            n1 = np.cross(P1, self.CR1 * R1 + self.CE1 * E1)
            n2 = np.cross(P2, self.CR2 * E1 + self.CE2 * E2)
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)

            ########### Project the velocity of rod in the normal direction ni
            V_n_p1 = self.Cp1 @ ((lV_P1 @ n1) * n1)
            V_n_p2 = self.Cp2 @ ((lV_P2 @ n2) * n2)
            dF1 = - 0.5 * 1000 * self.Cn1 @ (np.linalg.norm(V_n_p1) * V_n_p1) * np.linalg.norm(P1) * 2
            # print('dF1 ', dF1)
            dF2 = -0.5 * 1000 * self.Cn2 @ (np.linalg.norm(V_n_p2) * V_n_p2) * np.linalg.norm(P2) * 2
            # print('dF2 ', dF2)
            if left == True:
                R_bl = self.bending_fin(self.theta1_l)
                dW[:3, i] = R_bl @ (dF1 + dF2)
                dW[3:, i] = R_bl @ (np.cross(Q1, dF1) + np.cross(Q2, dF2))
            else:
                R_br = self.bending_fin(self.theta1_r)
                dW[:3, i] = R_br @ (dF1 + dF2)
                dW[3:, i] = R_br @ (np.cross(Q1, dF1) + np.cross(Q2, dF2))
        return dW

    ############机器人躯干的运动方程（求解机器人躯干当前的加速度）
    def motion_equation(self, t):
        # self.bending_fin(0.)
        '''
        psi_m = 40 / 180 * pi
        T = 2*pi / self.omega_bl
        #print(T)
        psi_b = 280 * pi / 360

        self.theta1_l = psi_m*sin(2 * pi * t / T + psi_b)
        self.theta1_r = - psi_m*sin(2 * pi * t / T + psi_b)
        self.d_theta1_l = 2 * pi / T * psi_m * cos(2 * pi * t / T + psi_b)
        self.d_theta1_r = -2 * pi / T * psi_m * cos(2 * pi * t / T + psi_b)
        '''
        self.rigid_body_kinematic()

        # print(t)
        WL, _ = integrate.fixed_quad(self.slice_hydrodynamic_force, 0, self.W, n=50, args=(t, True))
        # print('WL', WL)
        WR, _ = integrate.fixed_quad(self.slice_hydrodynamic_force, 0, self.W, n=50, args=(t, False))


        if np.linalg.norm(self.Vb) != 0:
            Vb_hat = self.Vb / np.linalg.norm(self.Vb)
            SB = Vb_hat @ (self.A @ Vb_hat)
            FB = -0.5 * 1000 * (np.linalg.norm(self.Vb) ** 2) * SB * self.CB @ Vb_hat
            print(FB)
        else:
            FB = np.zeros((3))
        

        MB = -self.Cw @ self.Omega_b
        acc = np.zeros((6))
        acc[:3] = (WL[:3] + WR[:3] + FB - np.cross(self.Omega_b, self.m * self.Vb)) / self.m
        # print('Fx', WL[0] + WR[0])
        M_b = 9.8 * self.m * np.cross(self.R_wb.T @ np.array([0, 0, -1]),
                                      np.array([0, 0, 0.001]))  # np.array([0,0,0.05])
        acc[3:] = self.inv_J @ (WL[3:] + WR[3:] + MB + M_b - np.cross(self.Omega_b, self.J @ self.Omega_b))
        # print(WL[3:] + WR[3:] + MB + M_b - np.cross(self.Omega_b, self.J @ self.Omega_b))
        return acc

    ########### 该函数用于测试在不同步态参数以及流体流速下的机器人受到的莫里森水动力
    def morison_parameter_test(self, omega_b, Vb, beta_l, beta_r, omega_bl, omega_br, psi_m, psi_b):
        self.omega_b, self.Vb = omega_b, Vb
        ####left right fin
        self.omega_bl = omega_bl
        self.omega_br = omega_br
        self.beta_l = beta_l  # 60 / 180 * pi
        self.beta_r = beta_r  # 60 / 180 * pi
        T = 2 * pi / self.omega_bl
        t = np.linspace(0, T, 51)
        F = np.zeros((6, t.size))
        F_left = np.zeros((6, t.size))
        F_right = np.zeros((6, t.size))
        # yaw fin
        for i in range(t.size):
            self.theta1_l = psi_m * sin(2 * pi * t[i] / T + psi_b)
            self.theta1_r = -psi_m * sin(2 * pi * t[i] / T + psi_b)
            self.d_theta1_l = 2 * pi / T * psi_m * cos(2 * pi * t[i] / T + psi_b)
            self.d_theta1_r = -2 * pi / T * psi_m * cos(2 * pi * t[i] / T + psi_b)
            self.rigid_body_kinematic()
            # print(t)
            WL, _ = integrate.fixed_quad(self.slice_hydrodynamic_force, 0, self.W, n=50, args=(t[i], True))
            # print('WL', WL)
            WR, _ = integrate.fixed_quad(self.slice_hydrodynamic_force, 0, self.W, n=50, args=(t[i], False))
            F[:, i] = WL + WR
            F_left[:, i] = WL
            F_right[:, i] = WR
        F_mean = np.zeros((6))
        for i in range(6):
            F_mean[i] = np.mean(F[i, :])
        return t, F, F_mean, F_left, F_right

    ############# 通过龙格库塔法求解躯干动力学方程
    def update_state(self, t, x, omega_bl, omega_br, beta_l, beta_r):
        self.omega_bl = omega_bl  # 0.8 * 2 * pi
        self.omega_br = omega_br
        xd = np.zeros((13))

        self.beta_l = beta_l
        self.beta_r = beta_r

        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        self.Vb = x[7:10]
        self.Omega_b = x[10:]
        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        self.R_wb, self.P_wb = R, x[:3]

        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd[3:7] = 0.5 * np.dot(G.T, self.Omega_b)  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, self.Vb)
        xd[7:] = self.motion_equation(t)
        return xd


    ########## 以下函数均用于绘制视频
    def draw(self, fig, t):
        '''
        psi_m = 40 / 180 * pi
        T = 2 * pi / self.omega_bl
        # print(T)
        psi_b = 280 * pi / 360
        # if t < 6 * T:
        #        psi_b = 280 * pi / 360
        # elif 6 * T < t < 8 * T:
        #        psi_b = 280 * pi / 360 - 200 * pi / 360 * (t - 6 * T) / (2 * T)
        # else:
        #        psi_b = 80 * pi / 360
        self.theta1_l = psi_m * sin(2 * pi * t / T + psi_b)
        self.theta1_r = -psi_m * sin(2 * pi * t / T + psi_b)
        self.beta_l = 60 / 180 * pi
        self.beta_r = 60 / 180 * pi


        if t < 6:
            self.beta_r = 60 / 180 * pi
        elif 6 <= t <= 7:
            self.beta_r = 60 / 180 * pi - (t - 6) * 120 / 180 * pi
        else:
            self.beta_r = -60 / 180 * pi
        '''
        R = self.R_wb
        P = self.P_wb
        # 3d bounding box corners
        l, h, w = 0.2, 0.35, 0.1
        x_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2]
        y_corners = [h / 2, h / 2, -h / 2, -h / 2, h / 2]
        z_corners = [w / 2, w / 2, w / 2, w / 2, w / 2]
        z_corners_ = [-w / 2, - w / 2, -w / 2, -w / 2, -w / 2]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners)
        for i in range(corners_3d.shape[1]):
            corners_3d[:, i] = corners_3d[:, i] + P
        ax = fig.gca(projection='3d', adjustable='box')
        ax.plot(corners_3d[0, :], corners_3d[1, :], corners_3d[2, :], color='red')
        corners = np.array([x_corners, y_corners, z_corners_], dtype=np.float32)
        corners_3d = np.dot(R, corners)
        for i in range(corners_3d.shape[1]):
            corners_3d[:, i] = corners_3d[:, i] + P
        ax.plot(corners_3d[0, :], corners_3d[1, :], corners_3d[2, :], color='red')
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, -l / 2, -l / 2, l / 2, l / 2]
        y_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners)
        for i in range(corners_3d.shape[1]):
            corners_3d[:, i] = corners_3d[:, i] + P
        ax.plot(corners_3d[0, :], corners_3d[1, :], corners_3d[2, :], color='red')
        self.R_bl = self.bending_fin(self.theta1_l)
        R_wl = self.R_wb @ self.R_bl
        P_wl = self.P_wb + self.R_wb @ self.P_bl

        self.complet_fin_position(beta=self.beta_l, omega_b=self.omega_bl, t=t, ax=ax, left=True, R=R_wl, P=P_wl)
        self.R_br = self.bending_fin(self.theta1_r)
        R_wr = self.R_wb @ self.R_br
        P_wr = self.P_wb + self.R_wb @ self.P_br

        self.complet_fin_position(beta=self.beta_r, omega_b=self.omega_br, t=t, ax=ax, left=False, R=R_wr, P=P_wr)
        # self.complet_fin_position(beta=self.beta_l, omega_b=self.omega_bl, t=t, ax=ax, left=True, R=np.eye(3), P=np.zeros((3)))
        ax.quiver(0, 0, 0, 0, 1, 0, length=0.1, color='blue')
        ax.quiver(0, 0, 0, 1, 0, 0, length=0.1, color='blue')
        ax.quiver(0, 0, 0, 0, 0, 1, length=0.1, color='blue')
        ax.quiver(self.P_wb[0], self.P_wb[1], self.P_wb[2], self.R_wb[0, 0], self.R_wb[1, 0], self.R_wb[2, 0],
                  length=0.1, color='red')
        ax.quiver(self.P_wb[0], self.P_wb[1], self.P_wb[2], self.R_wb[0, 1], self.R_wb[1, 1], self.R_wb[2, 1],
                  length=0.1, color='red')
        ax.quiver(self.P_wb[0], self.P_wb[1], self.P_wb[2], self.R_wb[0, 2], self.R_wb[1, 2], self.R_wb[2, 2],
                  length=0.1, color='red')
        ax.view_init(azim=-13, elev=-152)

        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.05, 1)
        ax.set_zlim(-0.4, 0.4)

    def projection(self, x, R, P):
        return R @ x + P

    def complet_fin_position(self, beta, omega_b, t, ax, left, R, P):
        '''
        :param beta: dephasage G1 G2
        :param omega_b: angular velocity
        :param t: time
        :return:
        '''

        W = 0.14  # fin width
        Ol = np.array([0, 0, 0])
        a = 0.024
        theta_bias_s = 2 * pi / 9
        if left == True:
            Oc = np.array([a * cos(theta_bias_s), W / 2, -a * sin(theta_bias_s)])  # (X_bs,Y_bs,Z_bs)
            Oc_ = np.array([a * cos(theta_bias_s), - W / 2, -a * sin(theta_bias_s)])
        else:
            Oc = np.array([-a * cos(theta_bias_s), W / 2, -a * sin(theta_bias_s)])  # (X_bs,Y_bs,Z_bs)
            Oc_ = np.array([-a * cos(theta_bias_s), - W / 2, -a * sin(theta_bias_s)])

        list_W = np.linspace(0, W, 2)
        l1_w = np.zeros((list_W.size, 3))
        l2_w = np.zeros((list_W.size, 3))
        L1 = 0.056
        L2 = 0.168
        k = 0.0045
        X_grid = np.zeros((list_W.size, 2))
        Y_grid = np.zeros((list_W.size, 2))
        Z_grid = np.zeros((list_W.size, 2))
        X2_grid = np.zeros((list_W.size, 2))
        Y2_grid = np.zeros((list_W.size, 2))
        Z2_grid = np.zeros((list_W.size, 2))

        for i in range(list_W.size):
            theta_b = omega_b * t + list_W[i] / W * beta
            theta_2, theta_3, _, _ = self.slice_fin_angle(list_W[i], t, left)
            if left == True:
                l1 = np.array([Oc[0] + L1 * cos(theta_2), Oc[1] - list_W[i], Oc[2] - L1 * sin(theta_2)])
                lenght = L2 - k * list_W[i] ** 2
                l2 = np.array([l1[0] + lenght * cos(theta_2 + theta_3), l1[1], l1[2] - lenght * sin(theta_2 + theta_3)])
                l1_w[i, :] = self.projection(l1, R, P)
                l2_w[i, :] = self.projection(l2, R, P)
                l0 = self.projection(np.array([a * cos(theta_bias_s), W / 2 - list_W[i], -a * sin(theta_bias_s)]), R, P)
            else:
                l1 = np.array([Oc[0] - L1 * cos(theta_2), Oc[1] - list_W[i], Oc[2] + L1 * sin(theta_2)])
                lenght = L2 - k * list_W[i] ** 2
                l2 = np.array([l1[0] - lenght * cos(theta_2 + theta_3), l1[1], l1[2] + lenght * sin(theta_2 + theta_3)])

                l1_w[i, :] = self.projection(l1, R, P)
                l2_w[i, :] = self.projection(l2, R, P)
                l0 = self.projection(np.array([-a * cos(theta_bias_s), W / 2 - list_W[i], -a * sin(theta_bias_s)]), R,
                                     P)

            X_grid[i, 0] = l1_w[i, 0]
            X_grid[i, 1] = l2_w[i, 0]
            Y_grid[i, 0] = l1_w[i, 1]
            Y_grid[i, 1] = l2_w[i, 1]
            Z_grid[i, 0] = l1_w[i, 2]
            Z_grid[i, 1] = l2_w[i, 2]

            X2_grid[i, 0] = l0[0]
            X2_grid[i, 1] = l1_w[i, 0]
            Y2_grid[i, 0] = l0[1]
            Y2_grid[i, 1] = l1_w[i, 1]
            Z2_grid[i, 0] = l0[2]
            Z2_grid[i, 1] = l1_w[i, 2]
            print(l1_w[i, 2])
        # ax = plt.subplot(111, projection='3d')
        # ax = fig.gca(projection='3d', adjustable='box')
        Oc_w = self.projection(Oc, R, P)
        Oc_w_ = self.projection(Oc_, R, P)
        Ol_w = self.projection(Ol, R, P)

        ax.plot_surface(X_grid, Y_grid, Z_grid, color='blue', alpha=0.3)
        ax.plot_surface(X2_grid, Y2_grid, Z2_grid, color='red', alpha=0.3)

        ax.plot(l1_w[:, 0], l1_w[:, 1], l1_w[:, 2], color='red')
        ax.plot(l2_w[:, 0], l2_w[:, 1], l2_w[:, 2], color='blue')
        ax.plot([Ol_w[0], Oc_w[0], l1_w[0, 0], l2_w[0, 0]], [Ol_w[1], Oc_w[1], l1_w[0, 1], l2_w[0, 1]],
                [Ol_w[2], Oc_w[2], l1_w[0, 2], l2_w[0, 2]],
                color='black')
        ax.plot([Ol_w[0], Oc_w_[0], l1_w[-1, 0], l2_w[-1, 0]], [Ol_w[1], Oc_w_[1], l1_w[-1, 1], l2_w[-1, 1]],
                [Ol_w[2], Oc_w_[2], l1_w[-1, 2], l2_w[-1, 2]],
                color='black')
        ax.scatter(l1_w[-1, 0], l1_w[-1, 1], l1_w[-1, 2],
                   color='black')

        ax.scatter(l1_w[0, 0], l1_w[0, 1], l1_w[0, 2],
                   color='black')
        ax.scatter(l2_w[-1, 0], l2_w[-1, 1], l2_w[-1, 2],
                   color='black')
        ax.scatter(l2_w[0, 0], l2_w[0, 1], l2_w[0, 2],
                   color='black')
        ax.scatter(Oc_w[0], Oc_w[1], Oc_w[2],
                   color='red')
        ax.scatter(Oc_w_[0], Oc_w_[1], Oc_w_[2],
                   color='blue')
        ax.scatter(Ol_w[0], Ol_w[1], Ol_w[2],
                   color='black')

start_time = datetime.now()
if __name__ == '__main__':


    manta = manta()

    manta.R_wb = np.eye(3)
    manta.P_wb = np.zeros((3))


    manta.Omega_b = np.zeros((3))
    manta.Vb = np.array([0, 0, 1])  # np.zeros((3))
    manta.beta_l = 60 / 180 * pi
    manta.beta_r = 60 / 180 * pi
    manta.d_theta1_l = 0.
    manta.d_theta1_r = 0.
    manta.theta1_l = 0.
    manta.theta1_r = 0.
    # manta.omega_bl = 0.8 * 2 * pi
    # manta.omega_br = 0.8 * 2 * pi
    time = np.linspace(0, 0.01, 2)
    #time = np.linspace(0, 10, 10001)
    from scipy.integrate import solve_ivp

    R = Rotation.from_matrix(manta.R_wb)
    quat = R.as_quat()
    Q1, Q2, Q3, Q4 = quat[3], quat[0], quat[1], quat[2]
    quat = np.array([Q1, Q2, Q3, Q4])
    x0 = np.zeros((13))
    x0[3:7] = quat
    x0[:3] = manta.P_wb
    x0[7:10] = manta.Vb
    x0[10:] = manta.Omega_b
    '''
    import cv2
    videowrite = cv2.VideoWriter('video_forward_swim_modified.mp4', -1, 25, (640, 480))
    # F=0.9,alpha0.1,tangage80.0_vid
    img_array = []

    for i in range(0,time.size,40):
        img = cv2.imread('./fig_forward_swim_modified/' + str(i) + '.jpg')
        img_array.append(img)
    for i in range(len(img_array)):
        print(i)
        videowrite.write(img_array[i])
    videowrite.release()
    print('ok')
    '''

    sol = solve_ivp(manta.update_state, [time[0], time[-1]], x0, t_eval=time, args=(pi, pi, 180/180*pi, 40/180*pi))
    state = sol.y
    list_euler = np.zeros((state.shape[1], 3))
    for i in range(state.shape[1]):
        quat = state[3:7, i]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        G = Rotation.from_matrix(R)
        euler = G.as_euler('ZYX', G)
        list_euler[i, :] = euler
    '''
    ########### 这部分用来绘制视频
    for i in range(0,time.size,40):
        fig = plt.figure()
        manta.draw(fig,time[i])
        quat = state[3:7, i]
        quat = quat / np.linalg.norm(quat)
        Q1 = quat[0];
        Q2 = quat[1];
        Q3 = quat[2];
        Q4 = quat[3];
        manta.R_wb = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        plt.savefig('./fig_forward_swim_modified/' + str(i) + '.jpg')
        manta.P_wb = state[:3, i]
        plt.close(fig)
    '''
    plt.figure()
    plt.title(r'$displacement of body$')
    plt.subplot(3, 1, 1)
    plt.plot(time, state[0, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('x(m)')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, state[1, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('y(m)')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, state[2, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('z(m)')
    plt.grid()
    plt.figure()
    plt.title(r'$Orientation of body$')
    plt.subplot(3, 1, 1)
    plt.plot(time, list_euler[:, 1], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('roll(deg)')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, list_euler[:, 2], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('pitch(deg)')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, list_euler[:, 0], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('yaw(deg)')
    plt.grid()
    plt.figure()
    plt.title(r'$Linear velocity$')
    plt.subplot(3, 1, 1)
    plt.plot(time, state[7, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('Vx(m/s)')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, state[8, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('Vy(m/s)')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, state[9, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel('Vz(m/s)')
    plt.grid()
    plt.figure()
    plt.title(r'$Angular velocity$')
    plt.subplot(3, 1, 1)
    plt.plot(time, state[11, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel(r'$\omega_x(rad/s)$')
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(time, state[10, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel(r'$\omega_y(rad/s)$')
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, state[12, :], color='black')
    plt.xlabel('time(s)')
    plt.ylabel(r'$\omega_z(rad/s)$')
    plt.grid()
    plt.show()
end_time = datetime.now()

# 计算并显示执行时间
execution_time = end_time - start_time
print(f"代码执行时间: {execution_time}")