import numpy as np
from math import *
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy import integrate, optimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import copy


class manta(object):
    def __init__(self):
        self.r2 = 0.185  # 0.112# 10cm 后杆长度
        self.D = 0.08  # 0.022# 5cm 前后电机横向距离
        self.L = 0.315  # 5cm 前后电机纵向距离
        self.q1l = 0.  # 左前杆角度
        self.q2l = 0.
        self.dq1l = 0.  # 左前杆速度
        self.dq2l = 0.
        self.ddq1l = 0.  # 左前杆加速度
        self.ddq2l = 0.
        self.q1r = 0.
        self.q2r = 0.
        self.dq1r = 0.  # 右前杆速度
        self.dq2r = 0.
        self.ddq1r = 0.  # 右前杆加速度
        self.ddq2r = 0.
        self.e_l = np.array([-self.L / 2, self.L, 0])  # 左鳍后杆铰接点在躯干坐标系中的位置
        self.e_r = np.array([-self.L / 2, -self.L, 0])  # 右鳍后杆铰接点在躯干坐标系中的位置
        self.b_l = np.array([self.L / 2, self.L + self.D, 0])  # 左鳍前杆铰接点在躯干坐标系中的位置
        self.b_r = np.array([self.L / 2, -self.L - self.D, 0])  # 右鳍前杆铰接点在躯干坐标系中的位置
        self.n = 10  # 30  # 鳍条数量
        self.q_r = np.zeros((3, self.n))  # 三自由度鳍条角度
        self.q_l = np.zeros((3, self.n))
        self.dq_r = np.zeros((3, self.n))  # 三自由度鳍条角速度
        self.dq_l = np.zeros((3, self.n))
        self.ddq_l = np.zeros((3, self.n))  # 三自由度鳍条角加速度
        self.ddq_r = np.zeros((3, self.n))

        self.Vb = np.zeros((3))  # 载体速度
        self.Omegab = np.zeros((3))  # 载体角速度
        self.dVb = np.zeros((3))  # 载体加速度
        self.dOmegab = np.zeros((3))  # 载体角加速度
        self.E = 3050E6  # 0.1E9  #43.7E6  橡胶杨氏模量
        self.G = 0.0003E9  # 橡胶剪切模量

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
        self.Omega_b = np.zeros((3))
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

        # self.Cn1 = np.diag([1.8, 0.5, 0.5])  # hydrodynamic coefficient of infinitesimal element 1
        # self.Cn1 = np.diag([1.8, 0.5, 0.5])*1
        self.Cn1 = np.array([[2.5, 0, 0], [0, 0.8, 0], [0, 0, 0.8]])  # np.diag([2.5, 0.8, 0.8])

        self.Cn2 = np.diag([0.5, 1.8, 0.5])  # hydrodynamic coefficient of infinitesimal element 2
        self.Cp1 = np.diag([1, 0.15, 1])  # np.diag([1, 0.15, 1])  # contribution of the velocity lV_Pi
        self.Cp2 = np.diag([0.15, 1, 1])  # np.diag([1, 0.15, 1])  # contribution of the velocity lV_Pi
        self.CR1 = 0.2
        self.CR2 = 0.3
        self.CE1 = 0.8
        self.CE2 = 0.7
        self.CB = np.diag([2, 4, 20]) * 0.1  # np.diag([0.2, 0.4, 1]) * 0.1 #0.0672  # np.diag((0.05,0.025,0.05))
        self.Cw = np.diag([0.015, 0.010, 0.020])  # np.diag((0.01,0.015,0.02))
        self.A = np.diag([0.8 * 0.4, 0.4 * 1, 1 * 0.8])  # np.diag([0.0298, 0.0167, 0.0577])
        print(self.A)
        self.m = 20

        self.J = np.diag(np.array([0.8 ** 2 + 0.4 ** 2, 1 ** 2 + 0.4 ** 2, 1 ** 2 + 0.8 ** 2]) * self.m / 12)
        print(self.J)
        self.inv_J = np.linalg.inv(self.J)

    def flexible_fin_determination(self):
        g = np.array([0, 0])
        a = np.array([self.L, 0])
        f = np.array([tan(30 / 180 * pi) * self.r2 * 0.5, self.r2 * 0.5])
        b = np.array([self.L + tan(-15 / 180 * pi) * self.r2, self.r2 * 0.5])
        e = np.array([tan(10 / 180 * pi) * self.r2, self.r2 * 0.5]) + f
        c = b + np.array([tan(-20 / 180 * pi) * self.r2, self.r2 * 0.5])
        d = c + np.array([tan(-25 / 180 * pi) * self.r2, self.r2 * 0.5])

        self.list_lenth = np.zeros((3, self.n))
        self.list_calibbration_angle = np.zeros((3, self.n))
        # 一级鳍条:

        x1 = np.linspace(g[0], a[0], self.n)
        y1 = np.linspace(g[1], a[1], self.n)
        x2 = np.linspace(f[0], b[0], self.n)
        y2 = np.linspace(f[1], b[1], self.n)
        x3 = np.linspace(e[0], c[0], self.n)
        y3 = np.linspace(e[1], c[1], self.n)

        for i in range(self.n):
            # plt.plot([x1[i],x2[i],x3[i],d[0]],[y1[i],y2[i],y3[i],d[1]])
            self.list_lenth[0, i] = sqrt((x1[i] - x2[i]) ** 2 + (y1[i] - y2[i]) ** 2)
            self.list_lenth[1, i] = sqrt((x2[i] - x3[i]) ** 2 + (y2[i] - y3[i]) ** 2)
            self.list_lenth[2, i] = sqrt((x3[i] - d[0]) ** 2 + (y3[i] - d[1]) ** 2)
            self.list_calibbration_angle[0, i] = -atan2(x2[i] - x1[i], y2[i] - y1[i])
            self.list_calibbration_angle[1, i] = -atan2(x3[i] - x2[i], y3[i] - y2[i]) - self.list_calibbration_angle[
                0, i]
            self.list_calibbration_angle[2, i] = -atan2(d[0] - x3[i], d[1] - y3[i]) - self.list_calibbration_angle[
                0, i] - self.list_calibbration_angle[1, i]

        '''
        plt.figure()
        plt.plot(x1,y1,color='red')
        plt.plot(x2, y2, color='red')
        plt.plot(x3, y3, color='red')
        plt.scatter(a[0],a[1])
        plt.scatter(b[0], b[1])
        plt.scatter(c[0],c[1])
        plt.scatter(d[0], d[1])
        plt.scatter(e[0], e[1])
        plt.scatter(f[0], f[1])
        plt.scatter(g[0], g[1])
        plt.plot([a[0],b[0],c[0],d[0],e[0],f[0],g[0]],[a[1],b[1],c[1],d[1],e[1],f[1],g[1]])
        plt.axis('equal')
        '''

        rho = 940  # 丁苯橡胶密度（SBR）
        h = 0.03  # 鳍厚度3cm
        self.d1 = np.mean(self.list_lenth[0, :])  # 一级平均长度
        self.d2 = np.mean(self.list_lenth[1, :])
        self.d3 = np.mean(self.list_lenth[2, :])
        w1 = np.linalg.norm(a - g) / self.n  # 一级平均宽度
        w2 = np.linalg.norm(b - f) / self.n
        w3 = np.linalg.norm(e - c) / self.n
        self.m1 = self.d1 * h * w1 * rho
        self.I1 = np.array([self.d1 ** 2 + h ** 2, w1 ** 2 + h ** 2, self.d1 ** 2 + w1 ** 2]) * self.m1 / 12
        self.M1 = np.zeros((6, 6))
        self.M1[:3, :3], self.M1[:3, 3:] = self.m1 * np.eye(3), - self.m1 * self.skew_symetric_matrix(
            np.array([0, self.d1 / 2, 0]))
        self.M1[3:, :3], self.M1[3:, 3:] = self.m1 * self.skew_symetric_matrix(np.array([0, self.d1 / 2, 0])), np.diag(
            self.I1) \
                                           - self.m1 * self.skew_symetric_matrix(
            np.array([0, self.d1 / 2, 0])) @ self.skew_symetric_matrix(np.array([0, self.d1 / 2, 0]))
        self.m2 = self.d2 * h * w2 * rho
        self.I2 = np.array([self.d2 ** 2 + h ** 2, w2 ** 2 + h ** 2, self.d2 ** 2 + w2 ** 2]) * self.m2 / 12
        self.M2 = np.zeros((6, 6))
        self.M2[:3, :3], self.M2[:3, 3:] = self.m2 * np.eye(3), - self.m2 * self.skew_symetric_matrix(
            np.array([0, self.d2 / 2, 0]))
        self.M2[3:, :3], self.M2[3:, 3:] = self.m2 * self.skew_symetric_matrix(np.array([0, self.d2 / 2, 0])), np.diag(
            self.I2) - self.m2 * self.skew_symetric_matrix(np.array([0, self.d2 / 2, 0])) @ self.skew_symetric_matrix(
            np.array([0, self.d2 / 2, 0]))

        self.m3 = self.d3 * h * w3 * rho
        self.I3 = np.array([self.d3 ** 2 + h ** 2, w3 ** 2 + h ** 2, self.d3 ** 2 + w3 ** 2]) * self.m3 / 12

        self.M3 = np.zeros((6, 6))
        self.M3[:3, :3], self.M3[:3, 3:] = self.m3 * np.eye(3), - self.m3 * self.skew_symetric_matrix(
            np.array([0, self.d3 / 2, 0]))
        self.M3[3:, :3], self.M3[3:, 3:] = self.m3 * self.skew_symetric_matrix(np.array([0, self.d3 / 2, 0])), np.diag(
            self.I3) - self.m3 * self.skew_symetric_matrix(np.array([0, self.d3 / 2, 0])) @ self.skew_symetric_matrix(
            np.array([0, self.d3 / 2, 0]))

        self.K1 = 3E-9 * self.E * self.I1[0] / self.d1

        self.K2 = 2E-9 * self.E * self.I2[0] / self.d2
        self.K3 = 1E-9 * self.E * self.I3[0] / self.d3
        print('固有周期1', sqrt(self.M1[3, 3] / self.K1) * 2 * pi, '固有周期2', sqrt(self.M2[3, 3] / self.K2) * 2 * pi,
              '固有周期3',
              sqrt(self.M3[3, 3] / self.K3) * 2 * pi)

        # print(self.E , self.I1[0] , d1, self.K1)
        self.K4 = self.G * (self.d1 ** 2 + h ** 2) * 1E-10
        print('刚度1', self.K1, '刚度2', self.K2, '刚度3', self.K3, '刚度4', self.K4)

    def bending_rotation_matrix(self, qb, qf):
        return np.array([[cos(qb), -sin(qb), 0], [sin(qb), cos(qb), 0], [0, 0, 1]]) @ \
            np.array([[1, 0, 0], [0, cos(qf), -sin(qf)], [0, sin(qf), cos(qf)]])

    def bending_rotation(self, qb):
        return np.array([[cos(qb), -sin(qb), 0], [sin(qb), cos(qb), 0], [0, 0, 1]])

    def roll_rotation(self, qr):
        return np.array([[1, 0, 0], [0, cos(qr), -sin(qr)], [0, sin(qr), cos(qr)]])

    def skew_symetric_matrix(self, M=np.array([])):
        M1 = M[0]
        M2 = M[1]
        M3 = M[2]
        Mx = np.array([[0, -M3, M2],
                       [M3, 0, -M1],
                       [-M2, M1, 0]])
        return Mx

    def config_matrix(self, Ri_im1, Pim1_1):
        Adg = np.zeros((6, 6))
        Adg[:3, :3], Adg[3:, 3:] = Ri_im1, Ri_im1
        Adg[:3, 3:] = Ri_im1 @ (self.skew_symetric_matrix(Pim1_1).T)

        return Adg

    def inverse_configuration(self, g):
        g_inv = np.eye(4)
        R_t = np.transpose(g[:3, :3])
        g_inv[:3, :3] = R_t;
        g_inv[:3, 3] = - np.dot(R_t, g[:3, 3])
        return g_inv

    def geometrical_model(self, qf):
        ###############################
        # 被动机构伴随表示
        ##############################
        list_adj = np.zeros((6, 6, 3, self.n))
        list_P03 = np.zeros((self.n))
        list_z = np.zeros((3, self.n))
        for i in range(self.n):
            R01 = self.bending_rotation_matrix(self.list_calibbration_angle[0, i], qf[0, i])
            P01 = np.zeros((3))
            R12 = self.bending_rotation_matrix(self.list_calibbration_angle[1, i], qf[1, i])
            P12 = np.array([0, self.d1, 0])  # np.array([0,self.list_lenth[0, i],0])
            R23 = self.bending_rotation_matrix(self.list_calibbration_angle[2, i], qf[2, i])
            P23 = np.array([0, self.d2, 0])  # np.array([0, self.list_lenth[1, i], 0])
            list_adj[:, :, 0, i] = self.config_matrix(R01.T, P01)
            list_adj[:, :, 1, i] = self.config_matrix(R12.T, P12)
            list_adj[:, :, 2, i] = self.config_matrix(R23.T, P23)
            # 用于计算末端弹性力
            list_P03[i] = (R01 @ R12 @ R23 @ np.array([0, self.d3, 0]) + R01 @ R12 @ P23 + R01 @ P12)[2]
            list_z[:, i] = R23.T @ R12.T @ R01.T @ np.array([0, 0, 1])
        return list_adj, list_P03, list_z

    def compute_fin_node(self, qf):
        # A = np.array([self.L, self.D/2 + self.r1 * cq1, self.r1 * sq1])
        A = np.array([self.L, 0, 0])
        # B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        B = np.array([0, 0, 0])
        l_AB = np.linalg.norm(A - B)

        l_AB = self.L
        gs = np.eye(4)

        l = np.linspace(0, l_AB, self.n)  # 鳍条AB间分段
        Coord = np.array([])
        index = [0, self.n - 1]
        for i in index:
            gs_ = gs @ np.array([[1, 0, 0, l[i]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            g01 = np.eye(4)
            g01[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[0, i], qf[0, 0])
            g01[:3, 3] = np.zeros((3))
            g12 = np.eye(4)
            g12[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[1, i], qf[0, 1])
            g12[:3, 3] = np.array([0, self.list_lenth[0, i], 0])  # np.array([0, self.list_lenth[0, i], 0])
            g23 = np.eye(4)
            g23[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[2, i], qf[0, 2])
            g23[:3, 3] = np.array([0, self.list_lenth[1, i], 0])  # np.array([0, self.list_lenth[1, i], 0])

            P1 = gs_ @ g01 @ np.array([0, 0, 0, 1])
            P2 = gs_ @ g01 @ np.array([0, self.list_lenth[0, i], 0, 1])
            P3 = gs_ @ g01 @ g12 @ np.array([0, self.list_lenth[1, i], 0, 1])
            P4 = gs_ @ g01 @ g12 @ g23 @ np.array([0, self.list_lenth[2, i], 0, 1])
            if i == 0:
                Coord = np.append(Coord, P2[:3])
                Coord = np.append(Coord, P3[:3])
                Coord = np.append(Coord, P4[:3])
            else:
                Coord = np.append(Coord, P3[:3])
                Coord = np.append(Coord, P2[:3])
        return Coord

    def draw_fin_rigid(self, ax, dis, left, time, color):
        dis = 0

        def create_cuboid_axes(ax, xlim, ylim, zlim):
            """创建一个长方体的3D坐标轴"""
            # 设置坐标轴范围
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            # 计算范围
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            z_range = zlim[1] - zlim[0]

            # 设置坐标轴比例，保持长方体形状
            ax.set_box_aspect([x_range, y_range, z_range])

        if left == True:
            q1, q2 = self.q1l, self.q2l
            qf = self.q_l
        else:
            q1, q2 = self.q1r, self.q2r
            qf = self.q_r
        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D

        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)

        # A = np.array([self.L, self.D/2 + self.r1 * cq1, self.r1 * sq1])
        A = np.array([self.L + dis, 0, 0])
        # B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        B = np.array([dis, 0, 0])
        l_AB = np.linalg.norm(A - B)

        theta = atan2(A[1] - B[1], self.L)

        beta = asin((B[2] - A[2]) / l_AB)
        # l_AB_ = self.L/cos(beta)/cos(theta)
        # print(l_AB-l_AB_)
        # R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]) @ np.array(
        #  [[cos(beta), 0, sin(beta)],
        #   [0, 1, 0],
        #    [-sin(beta), 0, cos(beta)]])
        R = np.eye(3)
        l_AB = self.L
        gs = np.eye(4)
        gs[0, 3] = dis
        l = np.linspace(0, l_AB, self.n)  # 鳍条AB间分段

        # gs[:3, :3], gs[:3, 3] = R, B

        # ax = fig.gca(projection='3d')
        # ax = fig.add_axes(Axes3D(fig))
        # ax.set_title('time:' + str(format(time, '.2f')) + 's', loc='center')
        # ax.scatter(0,0,0, color='black')
        # ax.scatter(self.L, self.D, 0, color='black')
        ax.scatter(B[0], B[1], B[2], color='black')
        # print(B, R)
        # ax.quiver(B[0], B[1], B[2], R[0, 0], R[1, 0], R[2, 0], length=0.1, color='black')
        ax.quiver(B[0], B[1], B[2], R[0, 1], R[1, 1], R[2, 1], length=0.1, color='black')
        # ax.quiver(B[0], B[1], B[2], R[0, 2], R[1, 2], R[2, 2], length=0.1, color='black')
        # ax.plot([0, B[0], A[0], self.L], [0, B[1], A[1], self.D], [0, B[2], A[2], 0], color='black')
        ax.plot([B[0], A[0]], [B[1], A[1]], [B[2], A[2]], color='black')
        index = [0, self.n - 1]
        corner_x = []
        corner_y = []
        corner_z = []
        for i in index:

            gs_ = gs @ np.array([[1, 0, 0, l[i]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            g01 = np.eye(4)
            g01[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[0, i], 0)
            g01[:3, 3] = np.zeros((3))
            g12 = np.eye(4)
            g12[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[1, i], 0)
            g12[:3, 3] = np.array([0, self.list_lenth[0, i], 0])  # np.array([0, self.d1, 0])
            g23 = np.eye(4)
            g23[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[2, i], 0)
            g23[:3, 3] = np.array([0, self.list_lenth[1, i], 0])  # np.array([0, self.d2, 0])

            P1 = gs_ @ g01 @ np.array([0, 0, 0, 1])
            P2 = gs_ @ g01 @ np.array([0, self.list_lenth[0, i], 0, 1])
            P3 = gs_ @ g01 @ g12 @ np.array([0, self.list_lenth[1, i], 0, 1])
            P4 = gs_ @ g01 @ g12 @ g23 @ np.array([0, self.list_lenth[2, i], 0, 1])
            if i == 0:
                corner_x.append(P1[0])
                corner_y.append(P1[1])
                corner_z.append(P1[2])
                corner_x.append(P2[0])
                corner_y.append(P2[1])
                corner_z.append(P2[2])
                corner_x.append(P3[0])
                corner_y.append(P3[1])
                corner_z.append(P3[2])
                corner_x.append(P4[0])
                corner_y.append(P4[1])
                corner_z.append(P4[2])
            else:
                corner_x.append(P3[0])
                corner_y.append(P3[1])
                corner_z.append(P3[2])
                corner_x.append(P2[0])
                corner_y.append(P2[1])
                corner_z.append(P2[2])
                corner_x.append(P1[0])
                corner_y.append(P1[1])
                corner_z.append(P1[2])
        R = self.bending_rotation_matrix(0, qf[0, 0])
        corner_x_rot = []
        corner_y_rot = []
        corner_z_rot = []
        for i in range(len(corner_x)):
            rot_corner = R @ np.array([corner_x[i], corner_y[i], corner_z[i]])
            corner_x_rot.append(rot_corner[0])
            corner_y_rot.append(rot_corner[1])
            corner_z_rot.append(rot_corner[2])
        ax.plot(corner_x_rot, corner_y_rot, corner_z_rot,
                color='black')
        ax.plot_trisurf(corner_x_rot, corner_y_rot, corner_z_rot, color=color, alpha=.5)

    def draw_fin(self, ax, dis, left, time, color):
        dis = 0

        def create_cuboid_axes(ax, xlim, ylim, zlim):
            """创建一个长方体的3D坐标轴"""
            # 设置坐标轴范围
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            # 计算范围
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            z_range = zlim[1] - zlim[0]

            # 设置坐标轴比例，保持长方体形状
            ax.set_box_aspect([x_range, y_range, z_range])

        if left == True:
            q1, q2 = self.q1l, self.q2l
            qf = self.q_l
        else:
            q1, q2 = self.q1r, self.q2r
            qf = self.q_r
        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D

        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)

        # A = np.array([self.L, self.D/2 + self.r1 * cq1, self.r1 * sq1])
        A = np.array([self.L + dis, 0, 0])
        # B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        B = np.array([dis, 0, 0])
        l_AB = np.linalg.norm(A - B)

        theta = atan2(A[1] - B[1], self.L)

        beta = asin((B[2] - A[2]) / l_AB)
        # l_AB_ = self.L/cos(beta)/cos(theta)
        # print(l_AB-l_AB_)
        # R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]) @ np.array(
        #  [[cos(beta), 0, sin(beta)],
        #   [0, 1, 0],
        #    [-sin(beta), 0, cos(beta)]])
        R = np.eye(3)
        l_AB = self.L
        gs = np.eye(4)
        gs[0, 3] = dis
        l = np.linspace(0, l_AB, self.n)  # 鳍条AB间分段

        # gs[:3, :3], gs[:3, 3] = R, B

        # ax = fig.gca(projection='3d')
        # ax = fig.add_axes(Axes3D(fig))
        # ax.set_title('time:' + str(format(time, '.2f')) + 's', loc='center')
        # ax.scatter(0,0,0, color='black')
        # ax.scatter(self.L, self.D, 0, color='black')
        ax.scatter(B[0], B[1], B[2], color='black')
        # print(B, R)
        # ax.quiver(B[0], B[1], B[2], R[0, 0], R[1, 0], R[2, 0], length=0.1, color='black')
        ax.quiver(B[0], B[1], B[2], R[0, 1], R[1, 1], R[2, 1], length=0.1, color='black')
        # ax.quiver(B[0], B[1], B[2], R[0, 2], R[1, 2], R[2, 2], length=0.1, color='black')
        # ax.plot([0, B[0], A[0], self.L], [0, B[1], A[1], self.D], [0, B[2], A[2], 0], color='black')
        ax.plot([B[0], A[0]], [B[1], A[1]], [B[2], A[2]], color='black')
        index = [0, self.n - 1]
        for i in index:
            gs_ = gs @ np.array([[1, 0, 0, l[i]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            g01 = np.eye(4)
            g01[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[0, i], qf[0, 0])
            g01[:3, 3] = np.zeros((3))
            g12 = np.eye(4)
            g12[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[1, i], qf[0, 1])
            g12[:3, 3] = np.array([0, self.list_lenth[0, i], 0])  # np.array([0, self.d1, 0])
            g23 = np.eye(4)
            g23[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[2, i], qf[0, 2])
            g23[:3, 3] = np.array([0, self.list_lenth[1, i], 0])  # np.array([0, self.d2, 0])

            P1 = gs_ @ g01 @ np.array([0, 0, 0, 1])
            P2 = gs_ @ g01 @ np.array([0, self.list_lenth[0, i], 0, 1])
            P3 = gs_ @ g01 @ g12 @ np.array([0, self.list_lenth[1, i], 0, 1])
            P4 = gs_ @ g01 @ g12 @ g23 @ np.array([0, self.list_lenth[2, i], 0, 1])
            ax.plot([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]],
                    color='black')
        list_l0_x = np.array([])
        list_l0_y = np.array([])
        list_l0_z = np.array([])
        list_l1_x = np.array([])
        list_l1_y = np.array([])
        list_l1_z = np.array([])
        list_l2_x = np.array([])
        list_l2_y = np.array([])
        list_l2_z = np.array([])
        list_l3_x = np.array([])
        list_l3_y = np.array([])
        list_l3_z = np.array([])

        for i in range(self.n):
            gs_ = gs @ np.array([[1, 0, 0, l[i]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            g01 = np.eye(4)
            g01[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[0, i], qf[0, 0])
            g01[:3, 3] = np.zeros((3))
            g12 = np.eye(4)
            g12[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[1, i], qf[0, 1])
            g12[:3, 3] = np.array([0, self.list_lenth[0, i], 0])  # np.array([0, self.list_lenth[0, i], 0])
            g23 = np.eye(4)
            g23[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[2, i], qf[0, 2])
            g23[:3, 3] = np.array([0, self.list_lenth[1, i], 0])  # np.array([0, self.list_lenth[1, i], 0])

            P1 = gs_ @ g01 @ np.array([0, 0, 0, 1])
            P2 = gs_ @ g01 @ np.array([0, self.list_lenth[0, i], 0, 1])
            P3 = gs_ @ g01 @ g12 @ np.array([0, self.list_lenth[1, i], 0, 1])
            P4 = gs_ @ g01 @ g12 @ g23 @ np.array([0, self.list_lenth[2, i], 0, 1])
            if i == 0:
                corner1 = P3
            elif i == self.n - 1:
                corner2 = P4
                corner3 = P3
            list_l0_x = np.append(list_l0_x, P1[0])
            list_l0_y = np.append(list_l0_y, P1[1])
            list_l0_z = np.append(list_l0_z, P1[2])
            list_l1_x = np.append(list_l1_x, P2[0])
            list_l1_y = np.append(list_l1_y, P2[1])
            list_l1_z = np.append(list_l1_z, P2[2])
            list_l2_x = np.append(list_l2_x, P3[0])
            list_l2_y = np.append(list_l2_y, P3[1])
            list_l2_z = np.append(list_l2_z, P3[2])
            # list_l3_x = np.append(list_l3_x, P4[0])
            # list_l3_y = np.append(list_l3_y, P4[1])
            # list_l3_z = np.append(list_l3_z, P4[2])
            # x = np.append(x, np.array([P1[0],P2[0],P3[0],P4[0]]))
            # y = np.append(y,np.array([P1[1],P2[1],P3[1],P4[1]]))
            # z = np.append(z, np.array([P1[2],P2[2],P3[2],P4[2]]))
            # ax.plot([P1[0],P2[0],P3[0],P4[0]],[P1[1],P2[1],P3[1],P4[1]],[P1[2],P2[2],P3[2],P4[2]],color='blue')
            # ax.plot([P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], [P1[2], P2[2], P3[2], P4[2]],
            #  color='black')
        ##ax.plot(list_l1_x,list_l1_y,list_l1_z ,color='red')
        ##ax.plot(list_l2_x, list_l2_y, list_l2_z, color='red')
        ##ax.plot(list_l3_x, list_l3_y, list_l3_z, color='red')
        x = np.array([])
        # x = np.append(x, list_l0_x)
        # x = np.append(x, np.flip(list_l1_x))
        # x = np.append(x, list_l2_x)
        # x = np.append(x, np.flip(list_l3_x))
        y = np.array([])
        # y= np.append(y, list_l0_y)
        # y = np.append(y, np.flip(list_l1_y))
        # y = np.append(y, list_l2_y)
        # y = np.append(y, np.flip(list_l3_y))
        z = np.array([])
        # z = np.append(z, list_l0_z)
        # z = np.append(z, np.flip(list_l1_z))
        # z = np.append(z, list_l2_z)
        # z = np.append(z, np.flip(list_l3_z))
        interpol_nb = 15
        list_pointx = [list_l0_x, list_l1_x, list_l2_x]
        list_pointy = [list_l0_y, list_l1_y, list_l2_y]
        list_pointz = [list_l0_z, list_l1_z, list_l2_z]
        ##list_pointx = [list_l0_x, list_l1_x, list_l2_x, list_l3_x]
        ##list_pointy = [list_l0_y, list_l1_y, list_l2_y, list_l3_y]
        ##list_pointz = [list_l0_z, list_l1_z, list_l2_z, list_l3_z]
        for i in range(len(list_pointx) - 1):
            x = np.array([])
            y = np.array([])
            z = np.array([])
            x = np.append(x, list_pointx[i])
            y = np.append(y, list_pointy[i])
            z = np.append(z, list_pointz[i])
            for k in range(interpol_nb - 2):
                if k % 2 == 1:
                    for j in range(list_l0_z.size):
                        x_interpol = np.linspace(list_pointx[i][j], list_pointx[i + 1][j], interpol_nb)
                        y_interpol = np.linspace(list_pointy[i][j], list_pointy[i + 1][j], interpol_nb)
                        z_interpol = np.linspace(list_pointz[i][j], list_pointz[i + 1][j], interpol_nb)

                        x = np.append(x, x_interpol[k + 1])
                        y = np.append(y, y_interpol[k + 1])
                        z = np.append(z, z_interpol[k + 1])
                else:
                    for j in reversed(range(list_l0_z.size)):
                        x_interpol = np.linspace(list_pointx[i][j], list_pointx[i + 1][j], interpol_nb)
                        y_interpol = np.linspace(list_pointy[i][j], list_pointy[i + 1][j], interpol_nb)
                        z_interpol = np.linspace(list_pointz[i][j], list_pointz[i + 1][j], interpol_nb)

                        x = np.append(x, x_interpol[k + 1])
                        y = np.append(y, y_interpol[k + 1])
                        z = np.append(z, z_interpol[k + 1])
            x = np.append(x, list_pointx[i + 1])
            y = np.append(y, list_pointy[i + 1])
            z = np.append(z, list_pointz[i + 1])
            # ax.plot(x, y, z, color=color)
            ax.plot_trisurf(x, y, z, color=color, alpha=1.)  # , edgecolor='none'
        ax.plot([corner1[0], corner2[0], corner3[0]], [corner1[1], corner2[1], corner3[1]],
                [corner1[2], corner2[2], corner3[2]],
                color='black')
        ax.plot_trisurf([corner1[0], corner2[0], corner3[0]], [corner1[1], corner2[1], corner3[1]],
                        [corner1[2], corner2[2], corner3[2]], color=color, alpha=1.)
        # x,y,z = plot_surface_arbitrary_domain(x,y,z, method= 'rbf', mask_method = 'alpha_shape')
        # ax.plot_surface(x, y, z,color='blue', alpha=0.8, edgecolor='none')
        # ax.plot_trisurf(x, y, z, color='blue', alpha=0.8, edgecolor='none')
        # ax.plot(x, y, z, color='blue')
        xlim = [-0., 0.3]
        ylim = [-0.0, 0.3]
        zlim = [-0.2, 0.2]
        create_cuboid_axes(ax, xlim, ylim, zlim)
        # ax.set_xlim(-3, 0.4)
        # ax.set_ylim(-0.1, 0.4)
        # ax.set_zlim(-0.2, 0.2)
        # ax.set_xlabel('x(m)')
        # ax.set_xticks([])
        # ax.set_ylabel('y(m)')
        # ax.set_zlabel('z(m)')
        ax.set_zticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_xticks([0, 0.1, 0.2])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        # ax.view_init(azim=96, elev=26,roll=-2)
        ax.view_init(azim=-10, elev=20)
        # ax.axis('equal')

        return theta, beta

    def draw_robot(self, fig, traj_des, traj, time):
        length, h, w = 0.35, 0.3, 0.1
        # ax = fig.gca(projection='3d')
        ax = fig.add_axes(Axes3D(fig))

        self.draw_one_side_fin(True, ax)

        self.draw_one_side_fin(False, ax)

        ax.plot(traj_des[0, :], traj_des[1, :], traj_des[2, :], linestyle='--', color='red')
        if traj.shape[1] != 0:
            ax.plot(traj[0, :], traj[1, :], traj[2, :], color='black')
        Rwb = self.R_wb
        P = self.P_wb

        x_corners = [length / 2, -length / 2, -length / 2, length / 2, length / 2]
        y_corners = [h / 2, h / 2, -h / 2, -h / 2, h / 2]
        z_corners = [w / 2, w / 2, w / 2, w / 2, w / 2]
        z_corners_ = [-w / 2, - w / 2, -w / 2, -w / 2, -w / 2]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(Rwb, corners)
        for i in range(corners_3d.shape[1]):
            corners_3d[:, i] = corners_3d[:, i] + P

        ax.plot(corners_3d[0, :], corners_3d[1, :], corners_3d[2, :], color='black')
        corners = np.array([x_corners, y_corners, z_corners_], dtype=np.float32)
        corners_3d = np.dot(Rwb, corners)
        for i in range(corners_3d.shape[1]):
            corners_3d[:, i] = corners_3d[:, i] + P
        ax.plot(corners_3d[0, :], corners_3d[1, :], corners_3d[2, :], color='black')
        x_corners = [length / 2, length / 2, -length / 2, -length / 2, -length / 2, -length / 2, length / 2, length / 2]
        y_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(Rwb, corners)
        for i in range(corners_3d.shape[1]):
            corners_3d[:, i] = corners_3d[:, i] + P
        ax.plot(corners_3d[0, :], corners_3d[1, :], corners_3d[2, :], color='black')

        ax.quiver(P[0], P[1], P[2], Rwb[0, 0], Rwb[1, 0], Rwb[2, 0], length=0.1, color='black')
        ax.quiver(P[0], P[1], P[2], Rwb[0, 1], Rwb[1, 1], Rwb[2, 1], length=0.1, color='black')
        ax.quiver(P[0], P[1], P[2], Rwb[0, 2], Rwb[1, 2], Rwb[2, 2], length=0.1, color='black')

        '''
        index = [0, self.n-1]
        for i in index:
            gs_ = gs @ np.array([[1, 0, 0, l[i]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            g01 = np.eye(4)
            g01[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[0, i], qf[0, i])
            g01[:3, 3] = np.zeros((3))
            g12 = np.eye(4)
            g12[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[1, i], qf[1, i])
            g12[:3, 3] =    np.array([0, self.list_lenth[0, i], 0])#np.array([0, self.d1, 0])
            g23 = np.eye(4)
            g23[:3, :3] = self.bending_rotation_matrix(self.list_calibbration_angle[2, i], qf[2, i])
            g23[:3, 3] =   np.array([0, self.list_lenth[1, i], 0])#np.array([0, self.d2, 0])

            P1 = gs_ @ g01 @ np.array([0, 0, 0, 1])
            P2 = gs_ @ g01 @ np.array([0, self.list_lenth[0, i], 0, 1])
            P3 = gs_ @ g01 @ g12 @ np.array([0, self.list_lenth[1, i], 0, 1])
            P4 = gs_ @ g01 @ g12 @ g23 @ np.array([0, self.list_lenth[2, i], 0, 1])
            ax.plot([P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], [P1[2], P2[2], P3[2], P4[2]],
                    color='black')
        '''

        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')
        ax.set_zlim(-1, 1)
        ax.set_title('time:' + str(format(time, '.2f')) + 's', loc='center')
        ax.view_init(azim=70, elev=60)

    def draw_one_side_fin(self, left, ax):
        length, h, w = 0.35, 0.3, 0.1
        if left == True:
            q1, q2 = self.q1l, self.q2l
            qf = self.q_l
        else:
            q1, q2 = self.q1r, self.q2r
            qf = self.q_r
        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D

        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)
        A_ = np.array([self.L, self.D / 2 + self.r1 * cq1, self.r1 * sq1])
        B_ = np.array([0, self.r2 * cq2, self.r2 * sq2])

        l_AB = np.linalg.norm(A_ - B_)

        theta = atan2(A_[1] - B_[1], self.L)

        beta = asin((B_[2] - A_[2]) / l_AB)
        if left == True:
            A = self.R_wb @ np.array([self.L / 2, self.D / 2 + self.r1 * cq1, self.r1 * sq1]) + self.P_wb
            B = self.R_wb @ np.array([-self.L / 2, self.r2 * cq2, self.r2 * sq2]) + self.P_wb
            O1 = self.R_wb @ np.array([self.L / 2, self.D, 0]) + self.P_wb
            O2 = self.R_wb @ np.array([-self.L / 2, 0.02, 0]) + self.P_wb
            R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]) @ np.array(
                [[cos(beta), 0, sin(beta)],
                 [0, 1, 0],
                 [-sin(beta), 0, cos(beta)]]) @ self.R_wb
        else:
            A = self.R_wb @ np.array([self.L / 2, -self.D / 2 - self.r1 * cq1, self.r1 * sq1]) + self.P_wb
            B = self.R_wb @ np.array([-self.L / 2, -self.r2 * cq2, self.r2 * sq2]) + self.P_wb
            O1 = self.R_wb @ np.array([self.L / 2, -self.D, 0]) + self.P_wb
            O2 = self.R_wb @ np.array([-self.L / 2, -0.02, 0]) + self.P_wb
            R = np.array([[cos(-theta), -sin(-theta), 0], [sin(-theta), cos(-theta), 0], [0, 0, 1]]) @ np.array(
                [[cos(beta), 0, sin(beta)],
                 [0, 1, 0],
                 [-sin(beta), 0, cos(beta)]]) @ self.R_wb
        ax.scatter(O1[0], O1[1], O1[2], color='blue')
        ax.scatter(O2[0], O2[1], O2[2], color='blue')
        ax.quiver(B[0], B[1], B[2], R[0, 0], R[1, 0], R[2, 0], length=0.1, color='blue')
        ax.quiver(B[0], B[1], B[2], R[0, 1], R[1, 1], R[2, 1], length=0.1, color='blue')
        ax.quiver(B[0], B[1], B[2], R[0, 2], R[1, 2], R[2, 2], length=0.1, color='blue')
        ax.plot([O2[0], B[0], A[0], O1[0]], [O2[1], B[1], A[1], O1[1]], [O2[2], B[2], A[2], O1[2]], color='blue')
        # l_AB_ = self.L/cos(beta)/cos(theta)
        # print(l_AB-l_AB_)

        l_AB = self.L
        gs = np.zeros((4, 4))
        l = np.linspace(0, l_AB, self.n)  # 鳍条AB间分段

        gs[:3, :3], gs[:3, 3] = R, B

        # ax = fig.gca(projection='3d')

        list_l1_x = []
        list_l1_y = []
        list_l1_z = []
        list_l2_x = []
        list_l2_y = []
        list_l2_z = []
        list_l3_x = []
        list_l3_y = []
        list_l3_z = []
        if left == True:
            sign = 1
        else:
            sign = -1
        for i in range(self.n):
            gs_ = gs @ np.array([[1, 0, 0, l[i]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            g01 = np.eye(4)
            g01[:3, :3] = self.bending_rotation_matrix(sign * self.list_calibbration_angle[0, i], sign * qf[i, 0])
            g01[:3, 3] = np.zeros((3))
            g12 = np.eye(4)
            g12[:3, :3] = self.bending_rotation_matrix(sign * self.list_calibbration_angle[1, i], sign * qf[i, 1])
            g12[:3, 3] = np.array([0, sign * self.d1, 0])  # np.array([0, self.list_lenth[0, i], 0])
            g23 = np.eye(4)
            g23[:3, :3] = self.bending_rotation_matrix(sign * self.list_calibbration_angle[2, i], sign * qf[i, 2])
            g23[:3, 3] = np.array([0, sign * self.d2, 0])  # np.array([0, self.list_lenth[1, i], 0])

            P1 = gs_ @ g01 @ np.array([0, 0, 0, 1])
            P2 = gs_ @ g01 @ np.array([0, sign * self.list_lenth[0, i], 0, 1])
            P3 = gs_ @ g01 @ g12 @ np.array([0, sign * self.list_lenth[1, i], 0, 1])
            P4 = gs_ @ g01 @ g12 @ g23 @ np.array([0, sign * self.list_lenth[2, i], 0, 1])
            '''
            list_l1_x.append(P2[0])
            list_l1_y.append(P2[1])
            list_l1_z.append(P2[2])
            list_l2_x.append(P3[0])
            list_l2_y.append(P3[1])
            list_l2_z.append(P3[2])
            list_l3_x.append(P4[0])
            list_l3_y.append(P4[1])
            list_l3_z.append(P4[2])
            '''
            ax.plot([P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], [P1[2], P2[2], P3[2], P4[2]],
                    color='blue')
            # ax.plot([P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], [P1[2], P2[2], P3[2], P4[2]],
            #       color='black')
        '''
        ax.plot(list_l1_x, list_l1_y, list_l1_z, color='red')
        ax.plot(list_l2_x, list_l2_y, list_l2_z, color='red')
        ax.plot(list_l3_x, list_l3_y, list_l3_z, color='red')
        '''
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')
        ax.set_zlim(-2, 2)
        ax.view_init(azim=70, elev=40)

        return theta, beta

    def velocity_transform(self, R, P, a, qd, V_n, Omega_n):
        V_np1 = R @ V_n + R @ np.cross(Omega_n, P)
        Omega_np1 = R @ Omega_n + a * qd
        return V_np1, Omega_np1

    def acceleration_transform(self, R, P, a, qd, qdd, dV_n, dOmega_n, V_np1, Omega_np1):
        dV_np1 = R @ dV_n + R @ np.cross(dOmega_n, P) + np.cross(V_np1, a * qd)
        dOmega_np1 = R @ dOmega_n + np.cross(Omega_np1, a * qd) + a * qdd
        return dV_np1, dOmega_np1

    def rotation_velocity_acceleration(self, dq1, dq2, ddq1, ddq2, cq1, cq2, sq1, sq2, cq1mq2, sq1mq2, theta):
        temp1 = - (dq1 * (self.r2 * cq1mq2 - self.D * cq1) * sq1 - dq2 * self.r2 * sq2)
        d_theta = temp1 * cos(theta) / self.L
        # dr1 = -self.D*sq2*cq1mq2*(dq1-dq2)/(sq1mq2**2)+self.D*cq2/sq1mq2*dq2\
        # +self.D/(sq1mq2**2)*sq1*(dq1-dq2)-self.D*cq1mq2/sq1mq2*cq1*dq1-self.r2*sq1mq2*(dq1-dq2)
        # d_theta = self.L**2/(self.L**2+(self.D+self.r1*cq1-self.r2*cq2)**2)*(dr1*cq1/self.L-self.r1*sq1*dq1/self.L+self.r2/self.L*sq2*dq2)

        dd_theta = - ((ddq1 * sq1 + dq1 ** 2 * cq1) * (self.r2 * cq1mq2 - self.D * cq1) + dq1 * sq1 * (
                -self.r2 * sq1mq2 * (
                dq1 - dq2) + self.D * sq1 * dq1) - ddq2 * self.r2 * sq2 - dq2 ** 2 * self.r2 * cq2) * \
                   cos(theta) / self.L + temp1 * sin(theta) / self.L * d_theta
        temp2 = - (dq1 * (self.r2 * cq1mq2 - self.D * cq1) * cq1 - dq2 * self.r2 * cq2)
        d_beta = temp2 * cos(theta) / self.L
        dd_beta = - ((ddq1 * cq1 - dq1 ** 2 * sq1) * (self.r2 * cq1mq2 - self.D * cq1) + dq1 * cq1 * (
                -self.r2 * sq1mq2 * (
                dq1 - dq2) + self.D * sq1 * dq1) - ddq2 * self.r2 * cq2 + dq2 ** 2 * self.r2 * sq2) * \
                  cos(theta) / self.L + temp2 * sin(theta) / self.L * d_theta
        return d_theta, d_beta, dd_theta, dd_beta

    '''
    def get_beta_zeta(self, eta_i, I, Aidqi):
        ###### Centrifuge/Corioli
        Vi = eta_i[:3];
        Oi = eta_i[3:]
        adj_eta = np.zeros((6, 6))
        adj_eta[:3, :3] = self.skew_symetric_matrix(Oi)
        adj_eta[3:, 3:] = self.skew_symetric_matrix(Oi)
        adj_eta[:3, 3:] = self.skew_symetric_matrix(Vi)
        # beta = -np.dot(adj_eta.T, np.dot(I, eta_i))
        beta = np.zeros((6))
        # beta[:3] = np.cross(eta_i[3:], -I[3:,:3] @ eta_i[3:])
        beta[3:] = np.cross(eta_i[3:], I[:3, :3] @ eta_i[3:])
        beta_ = -np.dot(adj_eta.T, np.dot(I, eta_i))
        zeta = adj_eta @ Aidqi
        return beta, zeta, beta_
    '''

    def get_zeta(self, eta_im1, Rjjm1, Pjm1j, dqj, a):
        Viim1 = Rjjm1 @ eta_im1[:3]
        Oiim1 = Rjjm1 @ eta_im1[3:]
        Piim1 = - Rjjm1 @ Pjm1j
        zeta = np.zeros((6))
        # zeta[:3] = Rjjm1 @ np.cross(Oim1, np.cross(Oim1, Pjm1j))
        # zeta[3:] = Rjjm1 @ np.cross(Oim1, dqj*a)
        # temp = np.cross(Oim1,Pjm1j)
        # zeta[:3] = Rjjm1@(np.cross(Oim1, temp))
        # zeta[3:] = np.cross(Rjjm1 @ Oim1, dqj*a)
        temp = Viim1 + np.cross(Piim1, Oiim1)
        zeta[:3] = np.cross(temp, dqj * a)
        zeta[3:] = np.cross(Oiim1, dqj * a)
        return zeta

    def get_beta(self, eta_i, I, J, S):

        ###### Centrifuge/Corioli
        Vi = eta_i[:3]
        Oi = eta_i[3:]
        beta = np.zeros((6))
        m = I[0, 0]

        Oi_hat = self.skew_symetric_matrix(Oi)
        S_hat = self.skew_symetric_matrix(S)
        beta[:3] = m * Oi_hat @ Oi_hat @ S
        beta[3:] = Oi_hat @ (J - m * S_hat @ S_hat) @ Oi

        return beta  # , zeta, beta_

    '''

    def get_beta(self, eta_i, J):
        Vi = eta_i[:3]
        Oi = eta_i[3:]
        beta = np.zeros((6))
        m = J[0,0]
        MS = J[3:,:3]
        I = J[3:,3:]
        beta[:3] = -np.cross(Oi, MS@Oi) + np.cross(Oi, m*Vi)
        beta[3:] = np.cross(Oi, I@Oi) + MS@(np.cross(Oi,Vi))
        return beta
    '''

    def active_mechanism_investigation(self, number_pend, left):
        if left == True:
            q1, q2 = self.q1l, self.q2l
            dq1, dq2 = self.dq1l, self.dq2l
            ddq1, ddq2 = self.ddq1l, self.ddq2l

            # qf[2,:] = 0.01*np.ones((self.n))
        else:
            q1, q2 = self.q1r, self.q2r
            dq1, dq2 = self.dq1r, self.dq2r
            ddq1, ddq2 = self.ddq1r, self.ddq2r

        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D
        ################################################
        # 主动机构速度传递
        ################################################
        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)
        cq1mq2, sq1mq2 = cos(q1 - q2), sin(q1 - q2)
        Vl = self.Vb + np.cross(self.Omegab, self.e_l)

        Omegal = self.Omegab

        A = np.array([self.L, self.D + self.r1 * cq1, self.r1 * sq1])
        B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        l_AB = np.linalg.norm(A - B)
        # print(l_AB)
        theta = atan2(A[1] - B[1], self.L)
        beta = asin((B[2] - A[2]) / l_AB)
        '''
        theta = atan2(self.D*(cq1-cq1**2)/self.L+self.r2*cq1mq2/(self.L*sq1mq2),1)
        if theta<0:
            theta = theta+pi/2
        elif theta>0:
            theta = theta - pi/2

        beta = -atan2( cos(theta)*cq1*(self.r2*sq1mq2-self.D*sq1)/self.L, 1)
        '''
        d_theta, d_beta, dd_theta, dd_beta = self.rotation_velocity_acceleration(dq1, dq2, ddq1, ddq2, cq1, cq2, sq1,
                                                                                 sq2, cq1mq2, sq1mq2, theta)

        Vo = Vl + np.cross(Omegal + np.array([1, 0, 0]) * dq2, B)
        Omegao = Omegal
        # if q1 != q2:
        #   d_r1 = self.D * (cq2 * dq2 - cq1 * dq1 * cq1mq2 + sq1 * sq1mq2 * (dq1 - dq2)) / sq1mq2 - \
        #    self.D * (sq2 - sq1 * cq1mq2) / sq1mq2 ** 2 * cq1mq2 * (dq1 - dq2)
        #  dd_r1 =
        # else:
        #   d_r1 = 0
        #  dd_r1 = 0
        # Vo = Vo + np.array([0, d_r1, 0])
        R1 = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
        R2 = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
        # 支架端点B在其坐标系Fo中的速度和角速度
        oV_theta = (R1.T @ Vo)
        oOmega_theta = (R1.T @ Omegao) + np.array([0, 0, 1]) * d_theta
        theta_V_beta = R2.T @ oV_theta
        theta_Omega_beta = R2.T @ oOmega_theta + np.array([0, 1, 0]) * d_beta
        #################################################
        # 主动机构加速度传递
        ################################################
        dVl = self.dVb + np.cross(self.dOmegab, self.e_l)
        dOmegal = self.dOmegab
        d_Vo = dVl + np.cross(dOmegal + np.array([1, 0, 0]) * ddq2, B) + \
               np.cross(Omegal + np.array([1, 0, 0]) * dq2, np.array([0, -self.r2 * sq2, self.r2 * cq2]) * dq2)
        d_Omegao = dOmegal

        d_oV_theta = R1.T @ d_Vo + np.cross(oV_theta, np.array([0, 0, 1]) * d_theta)
        d_oOmega_theta = R1.T @ d_Omegao + np.cross(oOmega_theta, np.array([0, 0, 1]) * d_theta) + np.array(
            [0, 0, 1]) * dd_theta
        d_theta_V_beta = R2.T @ d_oV_theta + np.cross(theta_V_beta, np.array([0, 1, 0]) * d_beta)
        d_theta_Omega_beta = R2.T @ d_oOmega_theta + np.cross(theta_Omega_beta,
                                                              np.array([0, 1, 0]) * d_beta) + np.array(
            [0, 1, 0]) * dd_beta
        l = np.linspace(0, self.L, self.n + 1)

        V_s = theta_V_beta + np.cross(theta_Omega_beta, np.array([l[number_pend], 0, 0]))
        dV_s = d_theta_V_beta + np.cross(d_theta_Omega_beta, np.array([l[number_pend], 0, 0]))
        return V_s, dV_s, theta, beta, d_theta, d_beta

    def pectoral_fin_dynamics(self, t, dt, qf, dqf, ddqf_old, left):
        if left == True:
            q1, q2 = self.q1l, self.q2l
            dq1, dq2 = self.dq1l, self.dq2l
            ddq1, ddq2 = self.ddq1l, self.ddq2l

            # qf[2,:] = 0.01*np.ones((self.n))
        else:
            q1, q2 = self.q1r, self.q2r
            dq1, dq2 = self.dq1r, self.dq2r
            ddq1, ddq2 = self.ddq1r, self.ddq2r

        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D
        ################################################
        # 主动机构速度传递
        ################################################
        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)
        cq1mq2, sq1mq2 = cos(q1 - q2), sin(q1 - q2)
        Vl = self.Vb + np.cross(self.Omegab, self.e_l)
        Omegal = self.Omegab

        A = np.array([self.L, self.D + self.r1 * cq1, self.r1 * sq1])
        B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        l_AB = np.linalg.norm(A - B)
        # print(l_AB)
        theta = atan2(A[1] - B[1], self.L)
        beta = asin((B[2] - A[2]) / l_AB)
        d_theta, d_beta, dd_theta, dd_beta = self.rotation_velocity_acceleration(dq1, dq2, ddq1, ddq2, cq1, cq2, sq1,
                                                                                 sq2, cq1mq2, sq1mq2, theta)

        Vo = Vl + np.cross(Omegal + np.array([1, 0, 0]) * dq2, B)

        Omegao = Omegal
        # if q1 != q2:
        #   d_r1 = self.D * (cq2 * dq2 - cq1 * dq1 * cq1mq2 + sq1 * sq1mq2 * (dq1 - dq2)) / sq1mq2 - \
        #    self.D * (sq2 - sq1 * cq1mq2) / sq1mq2 ** 2 * cq1mq2 * (dq1 - dq2)
        #  dd_r1 =
        # else:
        #   d_r1 = 0
        #  dd_r1 = 0
        # Vo = Vo + np.array([0, d_r1, 0])
        R1 = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
        R2 = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
        # 支架端点B在其坐标系Fo中的速度和角速度
        oV_theta = (R1.T @ Vo)
        oOmega_theta = (R1.T @ Omegao) + np.array([0, 0, 1]) * d_theta
        theta_V_beta = R2.T @ oV_theta
        theta_Omega_beta = R2.T @ oOmega_theta + np.array([0, 1, 0]) * d_beta
        #################################################
        # 主动机构加速度传递
        ################################################
        dVl = self.dVb + np.cross(self.dOmegab, self.e_l)
        dOmegal = self.dOmegab
        d_Vo = dVl + np.cross(dOmegal + np.array([1, 0, 0]) * ddq2, B) + \
               np.cross(Omegal + np.array([1, 0, 0]) * dq2, np.array([0, -self.r2 * sq2, self.r2 * cq2]) * dq2)
        d_Omegao = dOmegal

        d_oV_theta = R1.T @ d_Vo + np.cross(oV_theta, np.array([0, 0, 1]) * d_theta)
        d_oOmega_theta = R1.T @ d_Omegao + np.cross(oOmega_theta, np.array([0, 0, 1]) * d_theta) + np.array(
            [0, 0, 1]) * dd_theta
        d_theta_V_beta = R2.T @ d_oV_theta + np.cross(theta_V_beta, np.array([0, 1, 0]) * d_beta)
        d_theta_Omega_beta = R2.T @ d_oOmega_theta + np.cross(theta_Omega_beta,
                                                              np.array([0, 1, 0]) * d_beta) + np.array(
            [0, 1, 0]) * dd_beta

        ###################################################
        # 被动机构动力学
        ###################################################

        # l = np.linspace(0, l_AB, self.n+1)  # 鳍条AB间分段
        l = np.linspace(0, self.L, self.n + 1)
        # ds = l/self.n
        self.V_fin = np.zeros((3, self.n))
        self.dV_fin = np.zeros((3, self.n))
        self.Omega_fin = np.zeros((3, self.n))

        # list_adj, list_P03, list_z = self.geometrical_model(qf)
        a = np.array([0, 0, 0, 1, 0, 0])
        ddqf = np.zeros((3, self.n))
        Fl = np.zeros((6))  # 鳍受力

        Ad_gl0 = self.config_matrix(R2.T @ R1.T, B)
        # 梁参数
        m = 3
        L = 0.40
        d = L / m
        h = 0.004  # 0.03
        w = 0.015
        masse = self.d1 * h * w * 960
        I = np.array([d ** 2 + h ** 2, w ** 2 + h ** 2, d ** 2 + w ** 2]) * masse / 12
        K = self.E * I[0] * np.linspace(1, 1, m) * 10E-9
        # print('K',K*20/180*pi, masse*9.8*d/2)
        M = np.zeros((6, 6))
        M[:3, :3], M[:3, 3:] = masse * np.eye(3), - masse * self.skew_symetric_matrix(
            np.array([0, d / 2, 0]))
        M[3:, :3], M[3:, 3:] = masse * self.skew_symetric_matrix(np.array([0, d / 2, 0])), np.diag(
            I) - masse * self.skew_symetric_matrix(np.array([0, d / 2, 0])) @ self.skew_symetric_matrix(
            np.array([0, d / 2, 0]))

        list_Ad = np.zeros((6, 6, m))

        '''
        tf = 0.1
        poly = np.linalg.inv(np.array([[tf ** 3, tf ** 4, tf ** 5], [3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                                       [6 * tf, 12 * tf ** 2, 20 * tf ** 3]])) @ np.array([1, 0, 0])
        a3, a4, a5 = poly[0], poly[1], poly[2]
        amp = 1
        tf = 0.1
        freq = 0.1



        def vel(t):
            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                g = amp * sin(2 * pi * freq * t)
                velocity = f * g
            else:
                velocity = amp * sin(2 * pi * freq * t)
            # velocity = amp * sin(2 * pi / 0.1 * t)
            return velocity

        def acc(t):
            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                fp = 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2
                g = amp * sin(2 * pi * freq * t)
                gp = 2 * pi * freq * amp * cos(2 * pi * freq * t)
                acceleration = g * fp + gp * f
            else:
                acceleration = 2 * pi * freq * amp * cos(2 * pi / 0.1 * t)
            # acceleration = 2 * pi / 0.1 * amp * cos(2 * pi / 0.1 * t)
            return acceleration

        theta_V_beta = np.array([0, 0, vel(t)])
        d_theta_V_beta = np.array([0, 0, acc(t)])
        theta_Omega_beta = np.zeros((3))
        d_theta_Omega_beta = np.zeros((3))
        '''
        V_s = theta_V_beta
        dV_s = d_theta_V_beta

        list_beta = np.zeros((6, m))
        list_zeta = np.zeros((6, m))

        kappa = np.zeros((m))

        cinetic_energy = 0
        list_axis_z = np.zeros((3, m))
        potential_energy = 0

        V_s, theta_Omega_beta, dV_s, d_theta_Omega_beta = np.zeros((3)), np.zeros((3)), np.zeros((3)), np.zeros((3))

        for i in range(0, m, 1):
            # 速度传递
            R = np.array([[1, 0, 0], [0, cos(qf[i]), -sin(qf[i])], [0, sin(qf[i]), cos(qf[i])]])
            if i == 0:
                P = np.zeros((3))
            else:
                P = np.array([0, d, 0])
            Adgjjp1 = self.config_matrix(R.T, P)
            if i == 0:
                eta_im1 = np.append(V_s, theta_Omega_beta)
                eta = Adgjjp1 @ eta_im1 + dqf[i] * a
            else:
                eta_im1 = eta
                eta = Adgjjp1 @ eta_im1 + dqf[i] * a
            kappa[i] = 2 * tan(qf[i] / 2) / d
            # beta, zeta, beta_ = self.get_beta_zeta(eta, M, dqf[i] * a)

            ###################################################
            g_iim1 = np.zeros((4, 4))
            g_iim1[:3, :3] = R
            g_iim1[:3, 3] = P
            if i == 0:
                pos_CM = g_iim1 @ np.array([0, d / 2, 0, 1])
                list_axis_z[:, i] = R.T @ np.array([0, 0, -1])
            else:
                pos_CM = g_old @ g_iim1 @ np.array([0, d / 2, 0, 1])
                list_axis_z[:, i] = R.T @ list_axis_z[:, i - 1]
            g_old = g_iim1
            gravity = np.zeros((6))
            gravity[:3] = masse * 9.8 * list_axis_z[:, i]

            gravity[3:] = np.cross(np.array([0, d / 2, 0]), gravity[:3])

            potential_energy += 0.5 * K[i] * qf[i] ** 2 + masse * 9.8 * pos_CM[2]
            ####################################################

            beta = self.get_beta(eta, M, np.diag(I), np.array([0, d / 2, 0])) - gravity
            print('gravity', gravity[3])
            zeta = self.get_zeta(eta_im1, R.T, P, dqf[i], a[3:])

            temp = Viim1 + np.cross(Piim1, Oiim1)
            zeta[:3] = np.cross(temp, dqj * a)

            cinetic_energy += 0.5 * np.reshape(eta, (1, eta.size)) @ M @ eta
            list_Ad[:, :, i] = Adgjjp1
            list_beta[:, i] = beta
            list_zeta[:, i] = zeta

        A, B = np.zeros((m, m)), np.zeros((m))

        F = np.zeros((m))
        list_beta_star = np.zeros((6, m))
        list_M_star = np.zeros((6, 6, m))
        for j in range(m - 1, -1, -1):
            if j == m - 1:
                M_star = M
                list_M_star[:, :, j] = M_star
                beta_star = list_beta[:, j] + K[j] * qf[j] * a
                list_beta_star[:, j] = beta_star
            else:
                # print( (M_star@a)[3], (list_Ad[:, :, j+1].T@ M_star @ list_Ad[:, :, j+1]@a)[3])

                beta_star = list_beta[:, j] + K[j] * qf[j] * a + list_Ad[:, :, j + 1].T @ (
                            list_beta_star[:, j + 1] + list_M_star[:, :, j + 1] @ list_zeta[:, j + 1])
                list_beta_star[:, j] = beta_star
                M_star = M + list_Ad[:, :, j + 1].T @ list_M_star[:, :, j + 1] @ list_Ad[:, :, j + 1]
                list_M_star[:, :, j] = M_star
            Adg_ji = np.eye(6)
            # if j == 0:
            # A[j,j] = (M_star@a)[3]
            # B[j] = (-beta_star - M_star @ list_zeta[:,j] - M_star @ list_Ad[:,:,j] @ np.append(dV_s, d_theta_Omega_beta))[3]
            # F[j] = (-M_star @ list_Ad[:,:,j] @ np.append(dV_s, d_theta_Omega_beta))[3]

            Bj = -(list_beta_star[:, j])[3]
            for i in range(j, -1, -1):
                if i == j:
                    A[i, i] = (list_M_star[:, :, j] @ a)[3]
                    Bj += - (list_M_star[:, :, j] @ list_zeta[:, i])[3]  ###############i

                else:
                    Adg_ji = Adg_ji @ list_Ad[:, :, i + 1]
                    # print(M_star @ Adg_ji @ a)
                    A[j, i] = (list_M_star[:, :, j] @ Adg_ji @ a)[3]
                    A[i, j] = (Adg_ji.T @ list_M_star[:, :, j] @ a)[3]
                    Bj += - (list_M_star[:, :, j] @ Adg_ji @ list_zeta[:, i])[3]  ##############i

                if i == 0:
                    Adg_j0 = Adg_ji @ list_Ad[:, :, 0]
                    Bj += - (list_M_star[:, :, j] @ Adg_j0 @ np.append(dV_s, d_theta_Omega_beta))[3]
                    F[j] = - (list_M_star[:, :, j] @ Adg_ji @ np.append(dV_s, d_theta_Omega_beta))[3]
            B[j] = Bj
            # print(B_sym)

        # A = np.diag((np.linspace(0.01,0.001,30)))

        # list_energy +=  0.5 *np.reshape(qf, (1, qf.size))@ np.diag(K)@qf

        ddqf = np.linalg.inv(A) @ (B)  # - np.diag(K)@qf - np.diag(np.ones((qf.size)))*1E-2@dqf
        list_deta = np.zeros((6, m))
        list_verify_F = []

        for i in range(m):
            if i == 0:
                deta = list_Ad[:, :, i] @ np.append(dV_s, d_theta_Omega_beta) + list_zeta[:, i] + a * ddqf[i]
                list_deta[:, i] = deta
            else:
                deta = list_Ad[:, :, i] @ list_deta[:, i - 1] + list_zeta[:, i] + a * ddqf[i]
                list_deta[:, i] = deta

        for i in range(m - 1, -1, -1):
            deta = list_deta[:, i]
            # print('beta', M @ deta + list_beta[:,i])

            if i == m - 1:
                F_tot = (M @ deta + list_beta[:, i]) + (K[i] * qf[i] * a)

            else:
                F_tot = (M @ deta + list_beta[:, i]) + (K[i] * qf[i] * a) + (list_Ad[:, :, i + 1].T @ F_tot_old)

            F_tot_old = F_tot

            if i == m - 1:
                F_tot_ = (- K[i] * qf[i] * a)[3]
            else:
                F_tot_ = (- K[i] * qf[i] * a + list_Ad[:, :, i + 1].T @ (K[i + 1] * qf[i + 1] * a))[3]

            list_verify_F.append(F_tot[3])

        print('?', list_verify_F)
        # print('A',  (A))
        # print('qf',qf)
        # print('dqf', dqf)
        # print('ddqf',ddqf)
        #  更新被动连杆状态
        dqf_new = dqf + 0.5 * (ddqf + ddqf_old) * dt
        qf_new = qf + dqf * dt + dt ** 2 * (0.25 * ddqf + 0.25 * ddqf_old)
        work = F @ (qf_new - qf)
        # print(qf/pi*180)

        return qf_new, dqf_new, ddqf, F, cinetic_energy + potential_energy, work

    def pectoral_fin_dynamics_ode45(self, t, dt, list_qf, list_dqf, freq, Am1, phi, left, second_rate, last_rate):

        m = 3
        # self.Vb = np.array([0.,0,0])
        # self.Omegab = np.zeros((3))

        tf = 1 / freq
        poly = np.linalg.inv(np.array([[tf ** 3, tf ** 4, tf ** 5], [3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                                       [6 * tf, 12 * tf ** 2, 20 * tf ** 3]])) @ np.array([1, 0, 0])
        a3, a4, a5 = poly[0], poly[1], poly[2]
        # Am1 = 35 / 180 * pi  # 35 / 180 * pi
        Am2 = Am1 - asin(self.D ** 2 * sin(Am1) / self.r2)

        def pos_vec_acc(t):
            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                fp = 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2
                fpp = 20 * a5 * t ** 3 + 12 * a4 * t ** 2 + 6 * a3 * t
                g1 = Am1 * sin(2 * pi * freq * t + phi)
                g2 = Am2 * sin(2 * pi * freq * t)
                gp1 = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                gp2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                gpp1 = -(2 * pi * freq) ** 2 * Am1 * sin(2 * pi * freq * t + phi)
                gpp2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
                pos1 = f * g1
                pos2 = f * g2
                vec1 = f * gp1 + fp * g1
                vec2 = f * gp2 + fp * g2
                acc1 = f * gpp1 + fpp * g1 + 2 * fp * gp1
                acc2 = f * gpp2 + fpp * g2 + 2 * fp * gp2
            else:
                pos1 = Am1 * sin(2 * pi * freq * t + phi)
                pos2 = Am2 * sin(2 * pi * freq * t)
                vec1 = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                vec2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                acc1 = -(2 * pi * freq) ** 2 * Am1 * sin(2 * pi * freq * t + phi)
                acc2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
            return pos1, pos2, vec1, vec2, acc1, acc2

        def vel(t):

            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                g = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                velocity = f * g
                g2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                velocity2 = f * g2
            else:

                velocity = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                velocity2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
            # velocity = amp * sin(2 * pi / 0.1 * t)
            return velocity, velocity2

        def acc(t):

            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                fp = 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2
                g = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                gp = -(2 * pi * freq) ** 2 * Am1 * sin(2 * pi * freq * t + phi)
                g2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                gp2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
                acceleration = g * fp + gp * f
                acceleration2 = g2 * fp + gp2 * f
            else:

                acceleration = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                acceleration2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
            # acceleration = 2 * pi / 0.1 * amp * cos(2 * pi / 0.1 * t)
            return acceleration, acceleration2

        # self.q1l = Am1 * sin(2 * pi * freq * t + phi)
        # self.dq1l, self.dq2l  = vel(t)
        # self.ddq1l, self.ddq2l = acc(t)
        # self.q2l = Am2 * sin(2 * pi * freq * t)
        if left == True:
            self.q1l, self.q2l, self.dq1l, self.dq2l, self.ddq1l, self.ddq2l = pos_vec_acc(t)
        else:
            self.q1r, self.q2r, self.dq1r, self.dq2r, self.ddq1r, self.ddq2r = pos_vec_acc(t)
        if left == True:
            q1, q2 = self.q1l, self.q2l
            dq1, dq2 = self.dq1l, self.dq2l
            ddq1, ddq2 = self.ddq1l, self.ddq2l

            # qf[2,:] = 0.01*np.ones((self.n))
        else:
            q1, q2 = self.q1r, self.q2r
            dq1, dq2 = self.dq1r, self.dq2r
            ddq1, ddq2 = self.ddq1r, self.ddq2r

        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D
        ################################################
        # 主动机构速度传递
        ################################################
        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)
        cq1mq2, sq1mq2 = cos(q1 - q2), sin(q1 - q2)
        Vl = self.Vb + np.cross(self.Omegab, self.e_l)
        Omegal = self.Omegab

        A = np.array([self.L, self.D + self.r1 * cq1, self.r1 * sq1])
        B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        l_AB = np.linalg.norm(A - B)
        # print(l_AB)
        theta = atan2(A[1] - B[1], self.L)
        beta = asin((B[2] - A[2]) / l_AB)
        d_theta, d_beta, dd_theta, dd_beta = self.rotation_velocity_acceleration(dq1, dq2, ddq1, ddq2, cq1, cq2, sq1,
                                                                                 sq2, cq1mq2, sq1mq2, theta)

        Vo = Vl + np.cross(Omegal + np.array([1, 0, 0]) * dq2, B)

        Omegao = Omegal

        R1 = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
        R2 = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
        # 支架端点B在其坐标系Fo中的速度和角速度
        oV_theta = (R1.T @ Vo)
        oOmega_theta = (R1.T @ Omegao) + np.array([0, 0, 1]) * d_theta
        theta_V_beta = R2.T @ oV_theta
        theta_Omega_beta = R2.T @ oOmega_theta + np.array([0, 1, 0]) * d_beta
        #################################################
        # 主动机构加速度传递
        ################################################
        dVl = self.dVb + np.cross(self.dOmegab, self.e_l)
        dOmegal = self.dOmegab
        d_Vo = dVl + np.cross(dOmegal + np.array([1, 0, 0]) * ddq2, B) + \
               np.cross(Omegal + np.array([1, 0, 0]) * dq2, np.array([0, -self.r2 * sq2, self.r2 * cq2]) * dq2)
        d_Omegao = dOmegal

        d_oV_theta = R1.T @ d_Vo + np.cross(oV_theta, np.array([0, 0, 1]) * d_theta)
        d_oOmega_theta = R1.T @ d_Omegao + np.cross(oOmega_theta, np.array([0, 0, 1]) * d_theta) + np.array(
            [0, 0, 1]) * dd_theta
        d_theta_V_beta = R2.T @ d_oV_theta + np.cross(theta_V_beta, np.array([0, 1, 0]) * d_beta)
        d_theta_Omega_beta = R2.T @ d_oOmega_theta + np.cross(theta_Omega_beta,
                                                              np.array([0, 1, 0]) * d_beta) + np.array(
            [0, 1, 0]) * dd_beta

        ###################################################
        # 被动机构动力学
        ###################################################

        # l = np.linspace(0, l_AB, self.n+1)  # 鳍条AB间分段

        self.V_fin = np.zeros((3, self.n))
        self.dV_fin = np.zeros((3, self.n))
        self.Omega_fin = np.zeros((3, self.n))

        # list_adj, list_P03, list_z = self.geometrical_model(qf)

        gl0 = np.eye(4)
        gl0[:3, :3] = R1 @ R2
        gl0[:3, 3] = B

        L = 0.40
        span_lenth = 0.5
        d = span_lenth / m
        h = 0.03
        w = L / self.n

        masse = d * h * w * 960
        I = np.array([d ** 2 + h ** 2, w ** 2 + h ** 2, d ** 2 + w ** 2]) * masse / 12
        K = self.E * I[0] * np.array(
            [1, second_rate, last_rate]) * 2 * 10E-5  # self.E * I[0] * np.linspace(1, 0.1, m)  * 10E-6  # -4
        M = np.zeros((6, 6))
        M[:3, :3], M[:3, 3:] = masse * np.eye(3), - masse * self.skew_symetric_matrix(
            np.array([0, d / 2, 0]))
        M[3:, :3], M[3:, 3:] = masse * self.skew_symetric_matrix(np.array([0, d / 2, 0])), np.diag(
            I) - masse * self.skew_symetric_matrix(np.array([0, d / 2, 0])) @ self.skew_symetric_matrix(
            np.array([0, d / 2, 0]))

        l = np.linspace(0, self.L, self.n + 1)
        ds = self.L / self.n
        #####################################
        #### 莫里森水动力
        #####################################

        F_morison = np.zeros((self.n, m, 3))
        F = np.zeros((6))
        F1 = np.zeros((6))
        F2 = np.zeros((6))
        # 计算翼尖位移

        # self.tip_displacement = sin(list_qf[5,0]) * d + sin(list_qf[5,0] + list_qf[5,1]) * d + sin(
        #      list_qf[5,0] + list_qf[5,1] + list_qf[5, 2]) * d

        V_muulp_n = np.zeros((3))

        '''
        list_deph = np.linspace(0,20,m)*pi/180
        list_deph2 = np.linspace(-30, 0, self.n)*pi/180
        list_amplitude = np.array([20,30,40])*pi/180
        for i in range(self.n):
            for j in range(m):
                list_qf[i, j] = (1-0.5*i/self.n)*list_amplitude[j]*sin(2*pi*t+list_deph[j]+list_deph2[i])
                list_dqf[i,j] = 2*pi*(1-0.5*i/self.n)*list_amplitude[j]*cos(2*pi*t+list_deph[j]+list_deph2[i])
        '''
        nx = np.zeros((3, 3))
        list_g_li_old = np.zeros((m, 4, 4))
        for i in range(self.n):
            V = theta_V_beta + np.cross(theta_Omega_beta, np.array([l[i], 0, 0]))
            O = theta_Omega_beta

            for j in range(m):
                R = np.array([[1, 0, 0], [0, cos(list_qf[i, j]), -sin(list_qf[i, j])],
                              [0, sin(list_qf[i, j]), cos(list_qf[i, j])]])
                if j == 0:
                    P = np.zeros((3))
                else:
                    P = np.array([0, d, 0])
                g = np.eye(4)
                g[:3, :3], g[:3, 3] = R, P
                ####### 线段中点速度
                V = (V + R.T @ (np.cross(O, P)))
                O = R.T @ O + list_dqf[i, j] * np.array([1, 0, 0])
                V_middle = V + np.cross(O, np.array([0, d / 2, 0]))

                # if i != 0:
                # g_li_old = list_g_li_old[j,:,:]
                ####### 鳍条切线Rj,Ej
                '''
                if j == 0:
                    g_li = g
                    Rj = np.array([-1, 0, 0])
                else:
                    g_li = g_li @ g
                    temp = g_li_old @ np.array([0, 0, 0, 1])
                    temp2 = np.eye(4)
                    temp2[0, 3] = -ds
                    P2 = np.linalg.inv(g_li) @ (temp2 @ temp)
                    Rj = P2[:3]
                '''
                # temp = g_li_old @ np.append(np.array([0, d, 0]), 1)
                # temp2 = np.eye(4)
                # temp2[0, 3] = -ds
                # P3 = np.linalg.inv(g_li) @ (temp2 @ temp)
                # Ej = P3[:3] - np.array([0, d, 0])

                ########### 计算各个鳍条水动力
                # Rj, Ej = Rj / np.linalg.norm(Rj), Ej / np.linalg.norm(Ej)
                if j == 0:
                    Rj = np.array([-1, 0, 0])
                else:
                    Rj = np.array([-cos(list_qf[i, j]), 0, sin(list_qf[i, j])])
                Ej = np.array([-cos(list_qf[i, j]), 0, sin(list_qf[i, j])])
                # Rj, Ej = np.array([-1, 0, 0]), np.array([-1, 0, 0])
                nj = np.cross(np.array([-sin(list_qf[i, j]) * 0.0, d / 2, -cos(list_qf[i, j]) * 0.0]),
                              self.CR1 * Rj + self.CE1 * Ej)
                # print("Rj", Rj)
                # print("Ej", Ej)
                nj = nj / np.linalg.norm(nj)

                V_n = self.Cp1 @ ((V_middle @ nj) * nj)
                dF = -0.5 * 1000 * self.Cn1 @ (np.linalg.norm(V_n) * V_n) * d * ds

                F_morison[i, j, :] = dF
                if i == 5:
                    # print('?', dF)
                    V_muulp_n[j] = V_middle @ nj
                    nx[j, :] = dF
                # if i == 15 and j == 1:
                #   print(dF)
                # else:
                #    if j == 0:
                #        g_li = g
                #    else:
                #        g_li = g_li @ g
                # list_g_li_old[j,:,:] = copy.copy(g_li)

            # temp = np.eye(4)
            # temp[:3, 3] = np.array([l[i], 0., 0.])
            # gli = gl0 @ temp
            # gil = np.linalg.inv(gli)
            # Ad_gli = self.config_matrix(gil[:3, :3], gil[:3, 3])
        F_morison[0, :, :] = F_morison[1, :, :]

        '''
        V_15 = np.zeros((3,3))
        for i in range(self.n):
            V = theta_V_beta + np.cross(theta_Omega_beta, np.array([l[i], 0, 0]))
            O = theta_Omega_beta
            if i == 0:
                V_last = V
            if i == self.n-1:
                V_first = V
            for j in range(m):
                R = np.array([[1, 0, 0], [0, cos(list_qf[i,j]), -sin(list_qf[i,j])], [0, sin(list_qf[i,j]), cos(list_qf[i,j])]]) 
                if m == 0:
                    P = np.zeros((3))
                else:
                    P = np.array([0, d, 0])
                g = np.eye(4)
                g[:3, :3], g[:3, 3] = R, P
                ####### 线段中点速度
                V = (V+ R.T @ (np.cross(O, P)))
                O = R.T @ O + list_dqf[i, j] * np.array([1, 0, 0])
                V_middle = V + np.cross(O, P / 2)

                if i != 0:
                    ####### 鳍条切线Rj,Ej
                    if j == 0:
                        g_li = g
                        Rj = np.array([-1,0,0])
                    else:
                        g_li = g_li @ g
                        temp = g_li_old @ np.array([0,0,0,1])
                        temp2 = np.eye(4)
                        temp2[0,3] = -ds
                        P2 = np.linalg.inv(g_li) @ (temp2 @ temp)
                        Rj = P2[:3]
                    temp = g_li_old @ np.append(P,1)
                    temp2 = np.eye(4)
                    temp2[0, 3] = -ds
                    P3 = np.linalg.inv(g_li) @ (temp2 @ temp)
                    Ej = P3[:3] - P
                    ########### 计算各个鳍条水动力
                    nj = np.cross(P/2, self.CR1 * Rj + self.CE1 * Ej)
                    nj = nj / np.linalg.norm(nj)
                    V_n = self.Cp1 @ ((V_middle @ nj) * nj)

                    dF = -0.5 * 1000 * self.Cn1 @ (np.linalg.norm(V_n) * V_n) * d * ds

                    F_morison[i, j, :] = dF
                    if i == 15:
                        V_15[j,:] = Rj
                else:
                    if j == 0:
                        g_li = g
                    else:
                        g_li = g_li @ g
                g_li_old = copy.copy(g_li)

            temp = np.eye(4)
            temp[:3, 3] = np.array([l[i], 0., 0.])
            gli = gl0 @ temp
            gil = np.linalg.inv(gli)
            Ad_gli = self.config_matrix(gil[:3, :3], gil[:3, 3])
            #Ftot = np.zeros((6))
            #Ftot[:3] = F_morison[i, 0, :]+F_morison[i, 1, :]+F_morison[i, 2, :]
            #F = F + (Ad_gli.T) @ Ftot
        F_morison[0,:,:] = F_morison[1, :, :]
        '''

        list_qf_new = np.zeros((self.n, m))
        list_dqf_new = np.zeros((self.n, m))

        # F = np.zeros((6))
        time = np.array([t, t + dt])

        for i in range(self.n):
            V_s = theta_V_beta + np.cross(theta_Omega_beta, np.array([l[i], 0, 0]))
            dV_s = d_theta_V_beta + np.cross(d_theta_Omega_beta, np.array([l[i], 0, 0]))
            x = np.append(list_qf[i, :], list_dqf[i, :])
            sol = solve_ivp(self.one_fin_element_dynamics, [time[0], time[-1]], x, 'RK45', time,
                            args=[V_s, theta_Omega_beta, dV_s, d_theta_Omega_beta, d, M, K, I, F_morison[i, :, :], m])

            # print(sol.y[:m,-1])
            list_qf_new[i, :] = sol.y[:m, -1]
            list_dqf_new[i, :] = sol.y[m:, -1]
            dF = self.F0

            temp = np.eye(4)
            temp[:3, 3] = np.array([l[i], 0., 0.])
            gli = gl0 @ temp
            gil = np.linalg.inv(gli)
            Ad_gli = self.config_matrix(gil[:3, :3], gil[:3, 3])
            F = F - dF  # (Ad_gli.T) @
            F1 = F1 - self.F1
            F2 = F2 - (Ad_gli.T) @ dF
        return (F, F1, F2, list_qf_new, list_dqf_new)

    def one_fin_element_dynamics(self, t, x, V_s, theta_Omega_beta, dV_s, d_theta_Omega_beta, d, M, K, I, F_morison, m):
        qf = x[:m]
        dqf = x[m:]
        list_beta = np.zeros((6, m))
        list_zeta = np.zeros((6, m))
        list_Ad = np.zeros((6, 6, m))

        a = np.array([0, 0, 0, 1, 0, 0])
        for i in range(0, m, 1):
            # 速度传递
            R = np.array([[1, 0, 0], [0, cos(qf[i]), -sin(qf[i])], [0, sin(qf[i]), cos(qf[i])]])
            if i == 0:
                P = np.zeros((3))
            else:
                P = np.array([0, d, 0])
            Adgjjp1 = self.config_matrix(R.T, P)
            if i == 0:
                eta_im1 = np.append(V_s, theta_Omega_beta)
                eta = Adgjjp1 @ eta_im1 + dqf[i] * a
            else:
                eta_im1 = eta
                eta = Adgjjp1 @ eta_im1 + dqf[i] * a
            # beta, zeta, beta_ = self.get_beta_zeta(eta, M, dqf[i] * a)

            ###################################################
            '''
            g_iim1 = np.zeros((4, 4))
            g_iim1[:3, :3] = R
            g_iim1[:3, 3] = P
            if i == 0:
                pos_CM = g_iim1 @ np.array([0, d / 2, 0, 1])
                list_axis_z[:, i] = R.T @ np.array([0, 0, -1])
            else:
                pos_CM = g_old @ g_iim1 @ np.array([0, d / 2, 0, 1])
                list_axis_z[:, i] = R.T @ list_axis_z[:, i - 1]
            g_old = g_iim1
            gravity = np.zeros((6))
            gravity[:3] = masse * 9.8 * list_axis_z[:, i]

            gravity[3:] = np.cross(np.array([0, d / 2, 0]), gravity[:3])

            potential_energy += 0.5 * K[i] * qf[i] ** 2 + masse * 9.8 * pos_CM[2]
            '''
            ####################################################

            beta = self.get_beta(eta, M, np.diag(I), np.array([0, d / 2, 0]))  # - gravityself.get_beta(eta,M)#
            zeta = self.get_zeta(eta_im1, R.T, P, dqf[i], a[3:])

            if i == 0:
                Viim1 = R.T @ eta_im1[:3]
                Oiim1 = R.T @ eta_im1[3:]
                temp = Viim1 + np.cross(P, Oiim1)

            list_Ad[:, :, i] = Adgjjp1
            list_beta[:, i] = beta
            list_zeta[:, i] = zeta

        A, B = np.zeros((m, m)), np.zeros((m))

        F = np.zeros((m))
        list_beta_star = np.zeros((6, m))
        list_M_star = np.zeros((6, 6, m))
        for j in range(m - 1, -1, -1):
            if j == m - 1:
                M_star = M
                C = 1E-2  # 1E-2
                list_M_star[:, :, j] = M_star
                M_morison = np.cross(np.array([0, d / 2, 0]), F_morison[j, :])
                beta_star = list_beta[:, j] + 0.5 * K[j] * (1 / (cos(qf[j] / 2) ** 2) - 1) * np.sign(qf[j]) * a \
                            - np.append(F_morison[j, :], M_morison) + C * dqf[j]
                beta_star_ = - np.append(F_morison[j, :], M_morison)
                # beta_star = list_beta[:, j] + K[j] * qf[j] * a - np.append(F_morison[j,:], M_morison)
                list_beta_star[:, j] = beta_star

            else:
                if j == m - 2:
                    C = 1E-2  # 0.7E-2
                else:
                    C = 1E-2
                M_morison = np.cross(np.array([0, d / 2, 0]), F_morison[j, :])

                beta_star = list_beta[:, j] + 0.5 * K[j] * (1 / (cos(qf[j] / 2) ** 2) - 1) * np.sign(
                    qf[j]) * a + list_Ad[:, :, j + 1].T @ (
                                    list_beta_star[:, j + 1] + list_M_star[:, :, j + 1] @ list_zeta[:, j + 1]) \
                            - np.append(F_morison[j, :], M_morison) + C * dqf[j]
                beta_star_ += - np.append(F_morison[j, :], M_morison)
                # beta_star = list_beta[:, j] + K[j] * qf[j] * a + list_Ad[:, :, j + 1].T @ (
                #           list_beta_star[:, j + 1] + list_M_star[:, :, j + 1] @ list_zeta[:, j + 1])\
                #          - np.append(F_morison[j,:], M_morison)
                list_beta_star[:, j] = beta_star
                M_star = M + list_Ad[:, :, j + 1].T @ list_M_star[:, :, j + 1] @ list_Ad[:, :, j + 1]
                list_M_star[:, :, j] = M_star
            Adg_ji = np.eye(6)
            # if j == 0:
            # A[j,j] = (M_star@a)[3]
            # B[j] = (-beta_star - M_star @ list_zeta[:,j] - M_star @ list_Ad[:,:,j] @ np.append(dV_s, d_theta_Omega_beta))[3]
            # F[j] = (-M_star @ list_Ad[:,:,j] @ np.append(dV_s, d_theta_Omega_beta))[3]

            Bj = -(list_beta_star[:, j])[3]
            for i in range(j, -1, -1):
                if i == j:
                    A[i, i] = (list_M_star[:, :, j] @ a)[3]
                    Bj += - (list_M_star[:, :, j] @ list_zeta[:, i])[3]  ###############i
                else:
                    Adg_ji = Adg_ji @ list_Ad[:, :, i + 1]
                    # print(M_star @ Adg_ji @ a)
                    A[j, i] = (list_M_star[:, :, j] @ Adg_ji @ a)[3]
                    A[i, j] = (Adg_ji.T @ list_M_star[:, :, j] @ a)[3]
                    Bj += - (list_M_star[:, :, j] @ Adg_ji @ list_zeta[:, i])[3]  ##############i

                if i == 0:
                    Adg_j0 = Adg_ji @ list_Ad[:, :, 0]
                    Bj += - (list_M_star[:, :, j] @ Adg_j0 @ np.append(dV_s, d_theta_Omega_beta))[3]
            B[j] = Bj
            # print(B_sym)

        ddqf = np.linalg.inv(A) @ (B)  # - np.diag(K)@qf - np.diag(np.ones((qf.size)))*1E-2@dqf
        xd = np.zeros((2 * m))
        xd[:m] = dqf
        xd[m:] = ddqf
        self.F0 = list_beta_star[:, 0]  # beta_star_
        self.F1 = beta_star_
        # print(x)
        return xd

    def active_speed(self, t, freq, Am1):
        left = True
        m = 3
        self.Vb = np.array([0., 0, 0])
        self.Omegab = np.zeros((3))

        tf = 1 / freq
        poly = np.linalg.inv(np.array([[tf ** 3, tf ** 4, tf ** 5], [3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                                       [6 * tf, 12 * tf ** 2, 20 * tf ** 3]])) @ np.array([1, 0, 0])
        a3, a4, a5 = poly[0], poly[1], poly[2]
        # Am1 = 35 / 180 * pi  # 35 / 180 * pi
        Am2 = Am1 - asin(manta.D ** 2 * sin(Am1) / manta.r2)

        phi = 30 / 180 * pi

        def pos_vec_acc(t):
            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                fp = 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2
                fpp = 20 * a5 * t ** 3 + 12 * a4 * t ** 2 + 6 * a3 * t
                g1 = Am1 * sin(2 * pi * freq * t + phi)
                g2 = Am2 * sin(2 * pi * freq * t)
                gp1 = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                gp2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                gpp1 = -(2 * pi * freq) ** 2 * Am1 * sin(2 * pi * freq * t + phi)
                gpp2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
                pos1 = f * g1
                pos2 = f * g2
                vec1 = f * gp1 + fp * g1
                vec2 = f * gp2 + fp * g2
                acc1 = f * gpp1 + fpp * g1 + 2 * fp * gp1
                acc2 = f * gpp2 + fpp * g2 + 2 * fp * gp2
            else:
                pos1 = Am1 * sin(2 * pi * freq * t + phi)
                pos2 = Am2 * sin(2 * pi * freq * t)
                vec1 = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                vec2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                acc1 = -(2 * pi * freq) ** 2 * Am1 * sin(2 * pi * freq * t + phi)
                acc2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
            return pos1, pos2, vec1, vec2, acc1, acc2

        def vel(t):

            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                g = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                velocity = f * g
                g2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                velocity2 = f * g2
            else:

                velocity = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                velocity2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
            # velocity = amp * sin(2 * pi / 0.1 * t)
            return velocity, velocity2

        def acc(t):

            if t <= tf:
                f = a5 * t ** 5 + a4 * t ** 4 + a3 * t ** 3
                fp = 5 * a5 * t ** 4 + 4 * a4 * t ** 3 + 3 * a3 * t ** 2
                g = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                gp = -(2 * pi * freq) ** 2 * Am1 * sin(2 * pi * freq * t + phi)
                g2 = 2 * pi * freq * Am2 * cos(2 * pi * freq * t)
                gp2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
                acceleration = g * fp + gp * f
                acceleration2 = g2 * fp + gp2 * f
            else:

                acceleration = 2 * pi * freq * Am1 * cos(2 * pi * freq * t + phi)
                acceleration2 = -(2 * pi * freq) ** 2 * Am2 * sin(2 * pi * freq * t)
            # acceleration = 2 * pi / 0.1 * amp * cos(2 * pi / 0.1 * t)
            return acceleration, acceleration2

        # self.q1l = Am1 * sin(2 * pi * freq * t + phi)
        # self.dq1l, self.dq2l  = vel(t)
        # self.ddq1l, self.ddq2l = acc(t)
        # self.q2l = Am2 * sin(2 * pi * freq * t)
        self.q1l, self.q2l, self.dq1l, self.dq2l, self.ddq1l, self.ddq2l = pos_vec_acc(t)

        if left == True:
            q1, q2 = self.q1l, self.q2l
            dq1, dq2 = self.dq1l, self.dq2l
            ddq1, ddq2 = self.ddq1l, self.ddq2l

            # qf[2,:] = 0.01*np.ones((self.n))
        else:
            q1, q2 = self.q1r, self.q2r
            dq1, dq2 = self.dq1r, self.dq2r
            ddq1, ddq2 = self.ddq1r, self.ddq2r

        if q1 != q2:
            self.r1 = self.D * sin(q2) / sin(q1 - q2) + \
                      (self.r2 - self.D * sin(q1) / sin(q1 - q2)) * cos(q2 - q1)
        else:
            self.r1 = self.r2 - self.D
        ################################################
        # 主动机构速度传递
        ################################################
        sq1, cq1 = sin(q1), cos(q1)
        sq2, cq2 = sin(q2), cos(q2)
        cq1mq2, sq1mq2 = cos(q1 - q2), sin(q1 - q2)
        Vl = self.Vb + np.cross(self.Omegab, self.e_l)
        Omegal = self.Omegab

        A = np.array([self.L, self.D + self.r1 * cq1, self.r1 * sq1])
        B = np.array([0, self.r2 * cq2, self.r2 * sq2])
        l_AB = np.linalg.norm(A - B)
        # print(l_AB)
        theta = atan2(A[1] - B[1], self.L)
        beta = asin((B[2] - A[2]) / l_AB)
        d_theta, d_beta, dd_theta, dd_beta = self.rotation_velocity_acceleration(dq1, dq2, ddq1, ddq2, cq1, cq2, sq1,
                                                                                 sq2, cq1mq2, sq1mq2, theta)

        Vo = Vl + np.cross(Omegal + np.array([1, 0, 0]) * dq2, B)

        Omegao = Omegal

        R1 = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
        R2 = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

        # 支架端点B在其坐标系Fo中的速度和角速度
        oV_theta = (R1.T @ Vo)
        oOmega_theta = (R1.T @ Omegao) + np.array([0, 0, 1]) * d_theta
        theta_V_beta = R2.T @ oV_theta
        theta_Omega_beta = R2.T @ oOmega_theta + np.array([0, 1, 0]) * d_beta
        #################################################
        # 主动机构加速度传递
        ################################################
        dVl = self.dVb + np.cross(self.dOmegab, self.e_l)
        dOmegal = self.dOmegab
        d_Vo = dVl + np.cross(dOmegal + np.array([1, 0, 0]) * ddq2, B) + \
               np.cross(Omegal + np.array([1, 0, 0]) * dq2, np.array([0, -self.r2 * sq2, self.r2 * cq2]) * dq2)
        d_Omegao = dOmegal

        d_oV_theta = R1.T @ d_Vo + np.cross(oV_theta, np.array([0, 0, 1]) * d_theta)
        d_oOmega_theta = R1.T @ d_Omegao + np.cross(oOmega_theta, np.array([0, 0, 1]) * d_theta) + np.array(
            [0, 0, 1]) * dd_theta
        d_theta_V_beta = R2.T @ d_oV_theta + np.cross(theta_V_beta, np.array([0, 1, 0]) * d_beta)
        d_theta_Omega_beta = R2.T @ d_oOmega_theta + np.cross(theta_Omega_beta,
                                                              np.array([0, 1, 0]) * d_beta) + np.array(
            [0, 1, 0]) * dd_beta
        l = np.linspace(0, self.L, self.n + 1)

        V_s = theta_V_beta + np.cross(theta_Omega_beta, np.array([l[-1], 0, 0]))
        dV_s = d_theta_V_beta + np.cross(d_theta_Omega_beta, np.array([l[-1], 0, 0]))
        return cos(beta) / cos(theta), theta_V_beta, d_theta_V_beta, V_s, dV_s

    def draw(self, q, fig):
        m = 3
        L = 0.40
        d = L / m
        angle = 0
        list_x, list_y = [0], [0]
        for i in range(q.size):
            angle += q[i]
            list_x.append(list_x[-1] + cos(angle) * d)
            list_y.append(list_y[-1] + sin(angle) * d)

        plt.plot(list_x, list_y)
        plt.axis('equal')

    def motion_equation(self, t, x, dt, Am, phi, freq, qf, dqf, second_rate, last_rate):
        x[3:7] = x[3:7] / np.linalg.norm(x[3:7])

        F_l, _, __, qf_new, dqf_new = self.pectoral_fin_dynamics_ode45(t=t, dt=0.01, list_qf=qf,
                                                                       list_dqf=dqf, freq=freq, Am1=Am, phi=phi,
                                                                       second_rate=second_rate, last_rate=last_rate,
                                                                       left=True)
        # F_l = 10 * F_l
        F_r = copy.copy(F_l)
        F_r[1] = -F_r[1]
        F_r[3] = -F_r[3]
        F_r[5] = -F_r[5]
        F = F_r + F_l
        if np.linalg.norm(self.Vb) != 0:
            Vb_hat = self.Vb / np.linalg.norm(self.Vb)
            SB = Vb_hat @ (self.A @ Vb_hat)
            FB = -0.5 * 1000 * np.linalg.norm(self.Vb) ** 2 * SB * self.CB @ Vb_hat
        else:
            FB = np.zeros((3))

        MB = -self.Cw @ self.Omega_b
        acc = np.zeros((6))
        acc[:3] = (F[:3] + FB - np.cross(self.Omega_b, self.m * self.Vb)) / self.m
        # print('Fx', WL[0] + WR[0])
        ## 重力！！！！！
        M_b = 9.8 * self.m * np.cross(self.R_wb.T @ np.array([0, 0, -1]),
                                      np.array([0, 0, 1]))  # np.array([0,0,0.05])
        acc[3:] = self.inv_J @ (F[3:] + MB + M_b - np.cross(self.Omega_b, self.J @ self.Omega_b))
        quat = x[3:7]
        quat = quat / np.linalg.norm(quat)
        Q1, Q2, Q3, Q4 = quat[0], quat[1], quat[2], quat[3]

        self.Vb = x[7:10]
        self.Omega_b = x[10:]
        R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])
        self.R_wb, self.P_wb = R, x[:3]

        G = np.array([[-Q2, Q1, Q4, -Q3], [-Q3, -Q4, Q1, Q2], [-Q4, Q3, -Q2, Q1]])
        xd = np.zeros((13))
        xd[3:7] = 0.5 * np.dot(G.T, self.Omega_b)  # 0.5 * self.quaterion_product(quat,w)
        xd[:3] = np.dot(R, self.Vb)
        xd[7:] = acc
        x_new = x + xd * dt
        return x_new, qf_new, dqf_new, F

    def calculate_error(self, R_d_, R_):
        R = Rotation.from_matrix(R_)
        quat = R.as_quat()
        quat = quat / np.linalg.norm(quat)
        Q1, Q2, Q3, Q4 = quat[3], quat[0], quat[1], quat[2]
        quat = np.array([Q2, Q3, Q4])

        R_d = Rotation.from_matrix(R_d_)
        quat_d = R_d.as_quat()
        quat_d = quat_d / np.linalg.norm(quat_d)
        Q1d, Q2d, Q3d, Q4d = quat_d[3], quat_d[0], quat_d[1], quat_d[2]
        quat_d = np.array([Q2d, Q3d, Q4d])

        epsilon = Q1 * quat_d - Q1d * quat + np.cross(quat_d, quat)
        eta = Q1 * Q1d + quat @ quat_d

        Q1, Q2, Q3, Q4 = eta, epsilon[0], epsilon[1], epsilon[2]
        '''
        delta_R = np.array([[2 * (Q1 ** 2 + Q2 ** 2) - 1, 2 * (Q2 * Q3 - Q1 * Q4), 2 * (Q2 * Q4 + Q1 * Q3)],
                      [2 * (Q2 * Q3 + Q1 * Q4), 2 * (Q1 ** 2 + Q3 ** 2) - 1, 2 * (Q3 * Q4 - Q1 * Q2)],
                      [2 * (Q2 * Q4 - Q1 * Q3), 2 * (Q3 * Q4 + Q1 * Q2), 2 * (Q1 ** 2 + Q4 ** 2) - 1]])

        delta_R_ = np.transpose(R_) @ R_d_
        '''
        return epsilon



if __name__ == '__main__':
    manta = manta()
    manta.flexible_fin_determination()
    manta.n = 30  # 鳍条数量

    m = 3

    list_freq = [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]
    # list_second_rate = np.array([1.,0.95,0.8,0.7,.6,0.5,0.4])
    # list_rate = np.array([1.,0.8,0.6,0.4,.2,0.1,0.05])
    list_second_rate = np.array([1., 0.9, 0.8, 0.7, .6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    # list_second_rate = np.array([15, 10, 7])
    list_rate = np.array([1., 0.9, 0.8, 0.7, .6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    list_Fx = np.zeros((list_second_rate.size, list_rate.size))
    list_Mz = np.zeros((list_second_rate.size, list_rate.size))
    list_Fx1 = np.zeros((list_second_rate.size, list_rate.size))
    list_Mz1 = np.zeros((list_second_rate.size, list_rate.size))
    list_Fx2 = np.zeros((list_second_rate.size, list_rate.size))
    list_Mz2 = np.zeros((list_second_rate.size, list_rate.size))
    m = 3

    for j in range(list_second_rate.size):
        for k in range(list_rate.size):
            print(j, k)
            freq = 0.6  # list_freq[j]
            T = 1 / freq
            t = np.linspace(0, 3 * T, int(3 * T * 100 + 1))

            dqf = np.zeros((manta.n, m, t.size))
            qf = np.zeros((manta.n, m, t.size))
            F = np.zeros((6, t.size))
            F1 = np.zeros((6, t.size))
            F2 = np.zeros((6, t.size))
            list_lag_error = np.zeros((manta.n - 1, t.size - 1))
            list_tip_disp = np.zeros((t.size - 1))
            for i in range(t.size - 1):
                # F, list_qf_new, list_dqf_new, V_first, V_last, V_muulp_n, nx, self.test_term, self.test_term2
                F[:, i + 1], F1[:, i + 1], F2[:, i + 1], q, dqf[:, :,
                                                            i + 1] = manta.pectoral_fin_dynamics_ode45(
                    t=t[i], dt=0.01,
                    list_qf=qf[:, :, i],
                    list_dqf=dqf[:, :, i],
                    freq=freq, Am1=30 / 180 * pi,
                    phi=30 / 180 * pi, left=True, second_rate=list_second_rate[j], last_rate=list_rate[k])
                qf[:, :, i + 1] = q
            list_Fx[j, k] = np.mean(F[0, int(T * 100):])
            list_Mz[j, k] = np.mean(F[5, int(T * 100):])
            list_Fx1[j, k] = np.mean(F1[0, int(T * 100):])
            list_Mz1[j, k] = np.mean(F1[5, int(T * 100):])
            list_Fx2[j, k] = np.mean(F2[0, int(T * 100):])
            list_Mz2[j, k] = np.mean(F2[5, int(T * 100):])

    print('mean force morison+inertia', [list_Fx], [list_Mz])
    print('mean force morison', [list_Fx1], [list_Mz1])
    print('mean force morison+inertia+transformation', [list_Fx2], [list_Mz2])
    '''
    F = 0.35
    [[0.19144271 0.19391118 0.19663848 0.19964954 0.2029566  0.20653647
  0.21027735 0.21385476 0.21645519 0.21629569 0.21425564]
 [0.18912135 0.19151384 0.19415846 0.19707969 0.20028993 0.20376763
  0.20740664 0.21089875 0.21347508 0.2134641  0.21162515]
 [0.1864109  0.18871689 0.19126708 0.19408534 0.19718395 0.20054288
  0.20406175 0.20744975 0.20998668 0.21012367 0.20849779]
 [0.18323416 0.18544094 0.18788249 0.19058177 0.19355066 0.19677036
  0.20014635 0.20340639 0.20588342 0.2061635  0.20476145]
 [0.17950079 0.18159284 0.18390833 0.18646893 0.18928571 0.19234073
  0.19554528 0.19864687 0.20103645 0.2014482  0.20027809]
 [0.17510983 0.17706797 0.1792358  0.1816332  0.18426987 0.18712821
  0.19012513 0.19302914 0.1952944  0.19581677 0.19488192]
 [0.16995603 0.17175623 0.17374924 0.17595253 0.17837374 0.18099489
  0.18373829 0.18639474 0.18848706 0.189086   0.18838191]
 [0.16392854 0.1655402  0.16732383 0.16929377 0.17145475 0.17378773
  0.17622026 0.17856711 0.18042555 0.18105209 0.18056366]
 [0.1568133  0.15819799 0.1597291  0.16141714 0.16326331 0.16524718
  0.16730219 0.16926991 0.17082794 0.17142379 0.17112497]
 [0.1477525  0.14887275 0.15010976 0.15146999 0.15295121 0.15453243
  0.15615511 0.15769147 0.15890261 0.15941797 0.15927352]
 [0.14180385 0.14279183 0.14388171 0.14507806 0.14637723 0.14775842
  0.14916768 0.15049255 0.15153268 0.15199585 0.15190982]] [[-0.01595356 -0.01615926 -0.01638654 -0.01663746 -0.01691305 -0.01721137
  -0.01752311 -0.01782123 -0.01803793 -0.01802464 -0.01785464]
 [-0.01576011 -0.01595949 -0.01617987 -0.01642331 -0.01669083 -0.01698064
  -0.01728389 -0.0175749  -0.01778959 -0.01778867 -0.01763543]
 [-0.01553424 -0.01572641 -0.01593892 -0.01617378 -0.016432   -0.01671191
  -0.01700515 -0.01728748 -0.01749889 -0.01751031 -0.01737482]
 [-0.01526951 -0.01545341 -0.01565687 -0.01588181 -0.01612922 -0.01639753
  -0.01667886 -0.01695053 -0.01715695 -0.01718029 -0.01706345]
 [-0.0149584  -0.01513274 -0.01532569 -0.01553908 -0.01577381 -0.01602839
  -0.01629544 -0.01655391 -0.01675304 -0.01678735 -0.01668984]
 [-0.01459249 -0.01475566 -0.01493632 -0.0151361  -0.01535582 -0.01559402
  -0.01584376 -0.01608576 -0.01627453 -0.01631806 -0.01624016]
 [-0.014163   -0.01431302 -0.0144791  -0.01466271 -0.01486448 -0.01508291
  -0.01531152 -0.01553289 -0.01570725 -0.01575717 -0.01569849]
 [-0.01366071 -0.01379502 -0.01394365 -0.01410781 -0.0142879  -0.01448231
  -0.01468502 -0.01488059 -0.01503546 -0.01508767 -0.01504697]
 [-0.01306778 -0.01318317 -0.01331076 -0.01345143 -0.01360528 -0.0137706
  -0.01394185 -0.01410583 -0.01423566 -0.01428532 -0.01426041]
 [-0.01231271 -0.01240606 -0.01250915 -0.0126225  -0.01274593 -0.0128777
  -0.01301293 -0.01314096 -0.01324188 -0.01328483 -0.01327279]
 [-0.01181699 -0.01189932 -0.01199014 -0.01208984 -0.0121981  -0.0123132
  -0.01243064 -0.01254105 -0.01262772 -0.01266632 -0.01265915]]

  F = 0.45
  [[0.27809757 0.28235299 0.28704912 0.29222487 0.29789643 0.30402085
  0.31041541 0.31658336 0.32136771 0.32246401 0.32055742]
 [0.274564   0.27869339 0.28324963 0.28826962 0.29376794 0.29970166
  0.30589344 0.31186723 0.31652763 0.3177216  0.31602689]
 [0.27073454 0.27472268 0.27912173 0.28396622 0.2892686  0.29498554
  0.30094499 0.30669274 0.31119941 0.31247471 0.31099496]
 [0.26660426 0.27043285 0.27465401 0.27929932 0.28437858 0.28984753
  0.29553931 0.30102285 0.30533972 0.3066745  0.30541061]
 [0.26218629 0.26583304 0.26985105 0.27426843 0.27909164 0.28427498
  0.28965668 0.29483036 0.29891414 0.30028029 0.29923051]
 [0.25752008 0.2609572  0.26474072 0.26889465 0.27342147 0.27827345
  0.28329414 0.28810386 0.29190381 0.29326721 0.2924271 ]
 [0.25266647 0.25585826 0.25936743 0.26321322 0.26739355 0.27185841
  0.27645731 0.28084046 0.28429989 0.28562338 0.28498713]
 [0.24763209 0.25053153 0.25371451 0.25719507 0.26096636 0.26497657
  0.26908311 0.27297109 0.27603326 0.27728477 0.27684911]
 [0.24203869 0.24458487 0.24737573 0.2504202  0.25370733 0.25718541
  0.26072393 0.26405075 0.26667106 0.2678362  0.26760455]
 [0.23427171 0.23640388 0.23873796 0.24127848 0.24401208 0.24689031
  0.24980036 0.25252103 0.25467719 0.25574984 0.2557161 ]
 [0.22859639 0.23051679 0.23261757 0.2349013  0.23735385 0.23992913
  0.242524   0.2449429  0.24686688 0.24787856 0.24792335]] [[-0.0231748  -0.02352942 -0.02392076 -0.02435207 -0.0248247  -0.02533507
  -0.02586795 -0.02638195 -0.02678064 -0.026872   -0.02671312]
 [-0.02288033 -0.02322445 -0.02360414 -0.02402247 -0.02448066 -0.02497514
  -0.02549112 -0.02598894 -0.0263773  -0.0264768  -0.02633557]
 [-0.02256121 -0.02289356 -0.02326014 -0.02366385 -0.02410572 -0.02458213
  -0.02507875 -0.02555773 -0.02593328 -0.02603956 -0.02591625]
 [-0.02221702 -0.02253607 -0.02288783 -0.02327494 -0.02369821 -0.02415396
  -0.02462828 -0.02508524 -0.02544498 -0.02555621 -0.02545088]
 [-0.02184886 -0.02215275 -0.02248759 -0.0228557  -0.02325764 -0.02368958
  -0.02413806 -0.0245692  -0.02490951 -0.02502336 -0.02493588]
 [-0.02146001 -0.02174643 -0.02206173 -0.02240789 -0.02278512 -0.02318945
  -0.02360784 -0.02400865 -0.02432532 -0.02443893 -0.02436892]
 [-0.02105554 -0.02132152 -0.02161395 -0.02193444 -0.0222828  -0.02265487
  -0.02303811 -0.02340337 -0.02369166 -0.02380195 -0.02374893]
 [-0.02063601 -0.02087763 -0.02114288 -0.02143292 -0.0217472  -0.02208138
  -0.02242359 -0.02274759 -0.02300277 -0.02310706 -0.02307076]
 [-0.02016989 -0.02038207 -0.02061464 -0.02086835 -0.02114228 -0.02143212
  -0.02172699 -0.02200423 -0.02222259 -0.02231968 -0.02230038]
 [-0.01952264 -0.01970032 -0.01989483 -0.02010654 -0.02033434 -0.02057419
  -0.0208167  -0.02104342 -0.0212231  -0.02131249 -0.02130967]
 [-0.0190497  -0.01920973 -0.0193848  -0.01957511 -0.01977949 -0.01999409
  -0.02021033 -0.02041191 -0.02057224 -0.02065655 -0.02066028]]
    '''
    '''
    list_Fx = np.array([[0.05404186, 0.05528915, 0.05658973, 0.05778748, 0.0582213,  0.05751528,
      0.05671785],
     [0.09609591 ,0.09779342 ,0.09955872 ,0.1011519  ,0.10149724 ,0.10014253,
     0.09872048],
    [0.19144271,
    0.19415846,
    0.19718395,
    0.20014635,
    0.20103645,
    0.19875801,
    0.19632661],
    [0.27809757 ,0.28324963 ,0.2892686,  0.29553931, 0.29891414, 0.29685193,
     0.29418717],
    [0.31875078,
    0.32589703,
    0.3342189,
    0.34289818,
    0.34815152,
    0.34654505,
    0.34398663]])*30
    list_Mz =np.array( [[-0.00450349 ,- 0.00460743, - 0.00471581, - 0.00481562 ,- 0.00485177, - 0.00479294,
                   - 0.00472649],
                  [-0.00800799 ,- 0.00814945 ,- 0.00829656, - 0.00842933 ,- 0.0084581, - 0.00834521,
                   - 0.00822671],
                  [-0.01595356, - 0.01617987, - 0.016432, - 0.01667886 ,- 0.01675304, - 0.01656317,
                   - 0.01636055],
                  [-0.0231748, - 0.02360414, - 0.02410572, - 0.02462828, - 0.02490951 ,- 0.02473766,
                   - 0.0245156],
                  [-0.02656256, - 0.02715809 ,- 0.02785158, - 0.02857485, - 0.02901263 ,- 0.02887875,
                   - 0.02866555]])*30
    '''
    plt.subplot(1, 2, 1)
    for i in range(list_rate.size):
        plt.plot(list_freq, list_Fx[:, i],
                 label=r'$\gamma_1=$' + str(list_second_rate[i]) + r'$,\gamma_2=$' + str(list_rate[i]), marker='o')
    plt.xlabel('frequency(Hz)')
    plt.ylabel('mean thrust(N)')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    for i in range(list_rate.size):
        plt.plot(list_freq, list_Mz[:, i],
                 label=r'$\gamma_1=$' + str(list_second_rate[i]) + r'$,\gamma_2=$' + str(list_rate[i]), marker='o')
    plt.xlabel('frequency(Hz)')
    plt.ylabel('mean yaw moment(Nm)')
    plt.legend()
    plt.grid()
    plt.show()
    '''
    freq = 0.2
    T = 1 / freq
    t = np.linspace(0, 3 * T, int(3 * T * 100 + 1))
    list_Vy_forward = np.zeros((t.size))
    list_Vy_backward = np.zeros((t.size))
    list_dVy_forward = np.zeros((t.size))
    list_dVy_backward = np.zeros((t.size))
    list_dVy_forward_est = np.zeros((t.size))
    list_dVy_backward_est = np.zeros((t.size))
    F = np.zeros((6, t.size))
    list_beta_theta = []
    for i in range(t.size):
        beta_theta, V_backward, dV_backward, V_forward, dV_forward= manta.active_speed(t=t[i], freq=freq,Am1=20/180*pi)
        list_beta_theta.append(beta_theta)
        list_Vy_forward[i] = V_forward[2]
        list_Vy_backward[i] = V_backward[2]
        list_dVy_forward[i] = dV_forward[2]
        list_dVy_backward[i] = dV_backward[2]
    plt.figure()
    plt.plot(t, list_beta_theta, color='black')
    plt.xticks(ticks=[0, 0.333 * t[-1], 0.666 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$1$', r'$2$', r'$3$'], fontsize=12)
    plt.figure()
    plt.plot(t, list_Vy_forward, color='black')
    plt.plot(t, list_Vy_backward, color='red')
    for i in range(t.size):
        if i == 0:
            list_dVy_forward_est[i] = (list_Vy_forward[i+1]-list_Vy_forward[i])/0.01
            list_dVy_backward_est[i] = (list_Vy_backward[i+1] - list_Vy_backward[i])/0.01
        else:
            list_dVy_forward_est[i] = (list_Vy_forward[i] - list_Vy_forward[i-1]) / 0.01
            list_dVy_backward_est[i] = (list_Vy_backward[i] - list_Vy_backward[i-1]) / 0.01
    plt.figure()
    plt.plot(t, list_dVy_forward, color='black')
    plt.plot(t, list_dVy_backward, color='red')
    plt.plot(t, list_dVy_forward_est, color='black', linestyle='--')
    plt.plot(t, list_dVy_backward_est, color='red', linestyle='--')

    freq = 0.2
    T = 1 / freq
    t = np.linspace(0, 3 * T, int(3 * T * 100 + 1))
    list_Vy_forward = np.zeros((t.size))
    list_Vy_backward = np.zeros((t.size))
    list_dVy_forward = np.zeros((t.size))
    list_dVy_backward = np.zeros((t.size))
    list_dVy_forward_est = np.zeros((t.size))
    list_dVy_backward_est = np.zeros((t.size))
    F = np.zeros((6, t.size))
    list_beta_theta = []
    for i in range(t.size):
        beta_theta, V_backward, dV_backward, V_forward, dV_forward = manta.active_speed(t=t[i], freq=freq, Am1=40 / 180 * pi)
        list_beta_theta.append(beta_theta)
        list_Vy_forward[i] = V_forward[2]
        list_Vy_backward[i] = V_backward[2]
        list_dVy_forward[i] = dV_forward[2]
        list_dVy_backward[i] = dV_backward[2]
    plt.figure()
    plt.plot(t, list_beta_theta, color='black')
    plt.xticks(ticks=[0, 0.333 * t[-1], 0.666 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$1$', r'$2$', r'$3$'], fontsize=12)
    plt.figure()
    plt.plot(t, list_Vy_forward, color='black')

    plt.plot(t, list_Vy_backward, color='red')
    plt.xticks(ticks=[0, 0.333 * t[-1], 0.666 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$1$', r'$2$', r'$3$'], fontsize=12)
    for i in range(t.size):
        if i == 0:
            list_dVy_forward_est[i] = (list_Vy_forward[i + 1] - list_Vy_forward[i]) / 0.01
            list_dVy_backward_est[i] = (list_Vy_backward[i + 1] - list_Vy_backward[i]) / 0.01
        else:
            list_dVy_forward_est[i] = (list_Vy_forward[i] - list_Vy_forward[i - 1]) / 0.01
            list_dVy_backward_est[i] = (list_Vy_backward[i] - list_Vy_backward[i - 1]) / 0.01
    plt.figure()

    plt.plot(t, list_dVy_forward, color='black')
    plt.plot(t, list_dVy_backward, color='red')
    plt.plot(t, list_dVy_forward_est, color='black', linestyle='--')
    plt.plot(t, list_dVy_backward_est, color='red', linestyle='--')
    plt.xticks(ticks=[0, 0.333 * t[-1], 0.666 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$1$', r'$2$', r'$3$'], fontsize=12)
    plt.show()
    '''

    '''
    list_freq = [0.2,0.35,0.5]#[0.2,0.25,0.35,0.45,0.5]
    list_Am = np.array([20 / 180*pi, 25 / 180*pi, 30 / 180*pi, 35 / 180*pi, 40 / 180*pi])
    list_phi = np.array([-30 / 180 * pi, -10 / 180 * pi, 0 / 180 * pi, 10 / 180 * pi, 30 / 180 * pi])
    list_thrust = np.zeros((len(list_freq),len(list_Am)))
    list_thrust = np.array([[0.38537905, 0.56369553, 0.75292855, 0.94548319, 1.13677056],
       [0.77206662, 1.5527676 , 2.44307634, 3.43889137, 4.47754603],
       [2.34294704, 3.5243943, 4.19851556, 5.24615466, 6.69586157]])*1.5
    list_thrust_exp = np.array([[0.5, 0.75, 1, 1.9, 2.5],
                            [1.9,3.2,4.5,6,7],
                            [2.9,4.7, 6.1, 7.5, 8.5]])
    list_thrust2 = np.array([[0.80714781, 0.84550131, 0.82213233, 0.87577759, 0.75292855*1.5],
           [2.7682085, 2.95795601, 2.98390028, 3.551282, 2.44307634*1.5],
           [4.20363835, 4.61940236, 5.4410987, 5.6686378, 4.19851556*1.5]])
    list_thrust_exp2 = np.array([[-0.2, 0.3, 0.5, 0.9, 1],
                                [4.1, 4.2, 4.3, 4.45, 4.5],
                                [5, 5.3, 5.7, 6, 6.1]])
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(list_Am / pi * 180, list_thrust[0, :], marker='s', color='black', label='f=0.17Hz')
    plt.plot(list_Am / pi * 180, list_thrust[1, :], marker='o', color='black', label='f=0.34Hz')
    plt.plot(list_Am / pi * 180, list_thrust[2, :], marker='v', color='black', label='f=0.51Hz')
    plt.plot(list_Am / pi * 180, list_thrust_exp[0, :], marker='s', color='red', label='f=0.17Hz')
    plt.plot(list_Am / pi * 180, list_thrust_exp[1, :], marker='o', color='red', label='f=0.34Hz')
    plt.plot(list_Am / pi * 180, list_thrust_exp[2, :], marker='v', color='red', label='f=0.51Hz')
    # plt.plot(list_Am/pi*180, list_thrust[:, 3], marker='o', color='black', label='f=0.43Hz')
    # plt.plot(list_Am/pi*180, list_thrust[:, 4], marker='o', color='black', label='f=0.51Hz')
    plt.legend(ncol=3,fontsize=15)
    plt.xlabel('Flapping amplitude ' + 'Am(deg)', fontsize=18)
    plt.ylabel('The average thrust (N)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(list_phi / pi * 180, list_thrust2[0, :], marker='s', color='black', label='f=0.17Hz')
    plt.plot(list_phi / pi * 180, list_thrust2[1, :], marker='o', color='black', label='f=0.34Hz')
    plt.plot(list_phi / pi * 180, list_thrust2[2, :], marker='v', color='black', label='f=0.51Hz')
    plt.plot(list_phi / pi * 180, list_thrust_exp2[0, :], marker='s', color='red', label='f=0.17Hz')
    plt.plot(list_phi / pi * 180, list_thrust_exp2[1, :], marker='o', color='red', label='f=0.34Hz')
    plt.plot(list_phi / pi * 180, list_thrust_exp2[2, :], marker='v', color='red', label='f=0.51Hz')
    # plt.plot(list_Am/pi*180, list_thrust[:, 3], marker='o', color='black', label='f=0.43Hz')
    # plt.plot(list_Am/pi*180, list_thrust[:, 4], marker='o', color='black', label='f=0.51Hz')
    plt.legend(ncol=3,fontsize=15)
    plt.xlabel('Phase difference ' + r'$\phi$'+'(deg)', fontsize=18)
    plt.ylabel('The average thrust (N)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()

    list_freq = [0.4]#[0.4]  # [0.2,0.25,0.35,0.45,0.5]

    list_phi =np.array([40 / 180 * pi]) #np.array([30 / 180 * pi])
    for j in range(len(list_freq)):
        for k in range(list_phi.size):
            print(j,k)
            freq = list_freq[j]
            T = 1/freq
            t = np.linspace(0, 2*T, int(2*T*100+1))
            print(t[1]-t[0])
            dqf = np.zeros((manta.n, m, t.size))
            qf = np.zeros((manta.n, m, t.size))
            F = np.zeros((6, t.size))
            for i in range(t.size-1):
                F[:,i+1], q, dqf[:, :, i+1], _, _ = manta.pectoral_fin_dynamics_ode45(t=t[i], dt=0.01,list_qf=qf[:,:,i],
                                                                              list_dqf=dqf[:,:,i],freq = freq, Am1 = 30/180*pi, phi= list_phi[k])
                qf[:, :, i + 1] = q
                manta.q_l = q.T

            list_thrust[j, k] = np.mean(F[0, int(t.size / 2):]) * 2

            plt.figure()
            plt.subplot(6, 1, 1)
            plt.plot(t[1:], F[0, 1:])
            plt.plot(t[1:], np.mean(F[0, int(T * 100):]) * np.ones((t.size - 1)), linestyle='--')
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$F_x(N)$', fontsize=20)
            plt.xticks(ticks=[0, 0.333 * t[-1], 0.666 * t[-1], 1 * t[-1]],
                       labels=[r'$0$', r'$1$', r'$2$', r'$3$'], fontsize=15)
            plt.grid()
            plt.subplot(6, 1, 2)
            plt.plot(t[1:], F[1, 1:])
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$F_y(N)$', fontsize=20)
            plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                       labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=15)
            plt.grid()
            plt.subplot(6, 1, 3)
            plt.plot(t[1:], F[2, 1:])
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$F_z(N)$', fontsize=20)
            plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                       labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=15)
            plt.grid()
            plt.subplot(6, 1, 4)
            plt.plot(t[1:], F[3, 1:])
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$M_x(Nm)$', fontsize=20)
            plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                       labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=12)
            plt.grid()
            plt.subplot(6, 1, 5)
            plt.plot(t[1:], F[4, 1:])
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$M_y(Nm)$', fontsize=20)
            plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                       labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=12)
            plt.grid()
            plt.subplot(6, 1, 6)
            plt.plot(t[1:], F[5, 1:])
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$M_z(Nm)$', fontsize=20)
            plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                       labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=12)
            plt.grid()
            ylabel = [r'$q_1(deg)$',r'$q_2(deg)$',r'$q_3(deg)$']
            label =[r'$q_1^0$',r'$q_2^0$',r'$q_3^0$']
            label2 = [r'$q_1^{30}$', r'$q_2^{30}$', r'$q_3^{30}$']
            qmax = [10.2,13,17]
            qmax2 = [8.2,10.3,14.8]
            plt.figure()
            for i in range(3):
                plt.subplot(3, 1, i+1)
                plt.plot(t, qf[0, i, :] /np.max(qf[0, i, :])*qmax[i], color='black', label= label[i])
                max = np.max(qf[0, i, :] / pi * 180)
                index_max = np.argmax(qf[0, i, :] / pi * 180)
                plt.plot([t[index_max],t[index_max]], [-qmax[i],qmax[i]+2], color='black', linestyle = '--')
                plt.plot(t, qf[-1, i, :] /np.max(qf[-1, i, :])*qmax2[i], color='red', label=label2[i])

                index_max = np.argmax(qf[-1, i, :] / pi * 180)
                plt.plot([t[index_max], t[index_max]], [-qmax[i],qmax[i] + 2], color='red', linestyle='--')
                plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                           labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=15)
                plt.yticks(fontsize=12)
                plt.grid()
                plt.legend(fontsize=13, ncol=2)
                plt.xlabel(r'$t/T$', fontsize=18)
                plt.ylabel(ylabel[i], fontsize=18)


            plt.figure()
            plt.plot(t, qf[0, 0, :] / pi * 180, color='black', label=r'$q_1^{0}$')
            plt.plot(t, qf[0, 1, :] / pi * 180, color='red', label=r'$q_2^{0}$')
            plt.plot(t, qf[0, 2, :] / pi * 180, color='blue', label=r'$q_3^{0}$')
            plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], t[-1]],
                       labels=[r'$0$', r'$0.5$', r'$1$', r'$1.5$', r'$2$'], fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=13, ncol=2)
            plt.xlabel(r'$t/T$', fontsize=20)
            plt.ylabel(r'$q(deg)$', fontsize=20)
            plt.grid()
    plt.show()
    print('thrust', [list_thrust])

    plt.figure()
    plt.plot(list_phi/pi*180, list_thrust[0, :], marker='s', color = 'black', label='f=0.17Hz')
    plt.plot(list_phi/pi*180, list_thrust[1, :], marker='o', color='black', label='f=0.34Hz')
    plt.plot(list_phi/pi*180, list_thrust[2, :], marker='^', color='black', label='f=0.51Hz')
    #plt.plot(list_Am/pi*180, list_thrust[:, 3], marker='o', color='black', label='f=0.43Hz')
    #plt.plot(list_Am/pi*180, list_thrust[:, 4], marker='o', color='black', label='f=0.51Hz')
    plt.legend()
    plt.xlabel('Flapping amplitude'+r'$Am(^\circ)$', fontsize=18)
    plt.xlabel('The average thrust (N)', fontsize=18)
    plt.grid()
    plt.show()

    plt.figure()
    plt.subplot(6,1,1)
    plt.plot(t[1:], F[0,1:])
    plt.plot(t[1:], np.mean(F[0,int(t.size/2):])*np.ones((t.size-1)), linestyle = '--')
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$F_x(N)$', fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.subplot(6, 1, 2)
    plt.plot(t[1:], F[1, 1:])
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$F_y(N)$', fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.subplot(6, 1, 3)
    plt.plot(t[1:], F[2, 1:])
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$F_z(N)$', fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.subplot(6, 1, 4)
    plt.plot(t[1:], F[3, 1:])
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$M_x(Nm)$', fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.subplot(6, 1, 5)
    plt.plot(t[1:], F[4, 1:])
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$M_y(Nm)$', fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.subplot(6, 1, 6)
    plt.plot(t[1:], F[5, 1:])
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$M_z(Nm)$', fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()

    plt.figure()
    plt.subplot(3,1,1)


    plt.plot(t,qf[0,0,:]/pi*180, color = 'black', label=r'$q_1^0$')
    plt.plot(t, qf[-1,0, :] / pi * 180, color = 'red', label=r'$q_0^{30}$')
    plt.legend(fontsize=16)
    plt.xlabel(r'$t/T$',fontsize=20)
    plt.ylabel(r'$q_1(deg)$',fontsize=20)
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(t, qf[0, 1, :] / pi * 180, color = 'black', label=r'$q_2^0$')
    plt.plot(t, qf[-1, 1, :] / pi * 180, color = 'red', label=r'$q_2^{30}$')
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.grid()
    plt.legend(fontsize=16)
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$q_2(deg)$', fontsize=20)
    plt.subplot(3, 1, 3)
    plt.plot(t, qf[0, 2, :] / pi * 180, color = 'black', label=r'$q_3^0$')
    plt.plot(t, qf[-1, 2, :] / pi * 180, color = 'red', label=r'$q_3^{30}$')
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.legend(fontsize=16)
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$q_3(deg)$', fontsize=20)
    plt.grid()
    plt.figure()
    plt.plot(t, qf[0, 0, :] / pi * 180, color='black', label=r'$q_1^0$')
    plt.plot(t, qf[0, 1, :] / pi * 180, color='red', label=r'$q_2^0$')
    plt.plot(t, qf[0, 2, :] / pi * 180, color='blue', label=r'$q_3^0$')
    plt.xticks(ticks=[0, 0.25 * t[-1], 0.5 * t[-1], 0.75 * t[-1], 1 * t[-1]],
               labels=[r'$0$', r'$0.25$', r'$0.5$', r'$0.75$', r'$1$'], fontsize=12)
    plt.legend(fontsize=16)
    plt.xlabel(r'$t/T$', fontsize=20)
    plt.ylabel(r'$q_3(deg)$', fontsize=20)
    plt.grid()

    plt.show()




    list_ddtheta_est, list_ddbeta_est = np.zeros((3, t.size-1)),np.zeros((3, t.size-1))
    for i in range(1, t.size, 1):
        print('??', list_dtheta[:, i])
        list_ddtheta_est[:, i-1] = (list_dtheta[:, i]-list_dtheta[:, i-1])/0.001
        list_ddbeta_est[:, i-1] = (list_dbeta[:, i] - list_dbeta[:, i - 1]) / 0.001
    plt.figure()
    plt.subplot(6,1,1)
    plt.plot(list_ddbeta_est[0,:], color='black')
    plt.plot(list_ddbeta[0,:],color='red')
    plt.subplot(6, 1, 2)
    plt.plot(list_ddbeta_est[1, :], color='black')
    plt.plot(list_ddbeta[1, :], color='red')
    plt.subplot(6, 1, 3)
    plt.plot(list_ddbeta_est[2, :], color='black')
    plt.plot(list_ddbeta[2, :], color='red')
    plt.subplot(6, 1, 4)
    plt.plot(list_ddtheta_est[0,:], color='black')
    plt.plot(list_ddtheta[0,:], color='red')
    plt.subplot(6, 1, 5)
    plt.plot(list_ddtheta_est[1, :], color='black')
    plt.plot(list_ddtheta[1, :], color='red')
    plt.subplot(6, 1, 6)
    plt.plot(list_ddtheta_est[2, :], color='black')
    plt.plot(list_ddtheta[2, :], color='red')

    import cv2

    videowrite = cv2.VideoWriter('./New_manta_model/video_test_fin.mp4', -1, 20, (640, 480))
    # F=0.9,alpha0.1,tangage80.0_vid
    img_array = []

    for i in range(0, t.size, 20):
        img = cv2.imread('./New_manta_model/fin-test/' + str(i) + '.jpg')
        img_array.append(img)
    for i in range(len(img_array)):
        print(i)
        videowrite.write(img_array[i])
    videowrite.release()
    '''
