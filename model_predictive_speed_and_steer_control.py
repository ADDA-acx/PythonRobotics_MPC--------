"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from utils.angle import angle_mod

from CubicSpline import cubic_spline_planner

NX = 4  # 状态变量数目：位置x, 位置y, 速度v, 偏航角yaw
NU = 2  # 控制输入数目：加速度accel, 转向角steer


# MPC参数
R = np.diag([0.01, 0.01])  # 输入代价权重矩阵
Rd = np.diag([0.01, 1.0])  # 输入差值代价权重矩阵
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # 状态代价权重矩阵
Qf = Q  # 最终状态代价矩阵
GOAL_DIS = 1.5  # 目标距离
STOP_SPEED = 0.5 / 3.6  # 停止速度
T = 5  # 时间预测长度
MAX_TIME = 500.0  # 最大模拟时间


# 迭代参数
MAX_ITER = 3  # 最大迭代次数
DU_TH = 0.1  # 迭代结束条件

TARGET_SPEED = 10.0 / 3.6  # 目标速度
N_IND_SEARCH = 10  # 搜索索引数

DT = 0.2  # 时间步长

# 车辆参数
LENGTH = 4.5  # 车辆长度
WIDTH = 2.0  # 车辆宽度
BACKTOWHEEL = 1.0  # 后轮到车尾的距离
WHEEL_LEN = 0.3  # 车轮长度
WHEEL_WIDTH = 0.2  # 车轮宽度
TREAD = 0.7  # 轮距
WB = 2.5  # 轴距

MAX_STEER = np.deg2rad(45.0)  # 最大转向角
MAX_DSTEER = np.deg2rad(30.0)  # 最大转向速度
MAX_SPEED = 55.0 / 3.6  # 最大速度
MIN_SPEED = -20.0 / 3.6  # 最小速度
MAX_ACCEL = 1.0  # 最大加速度

show_animation = True  # 是否显示动画

is_back = True  # 是否倒车

# 定义车辆状态类用于记录当前车辆的位置、速度和偏航角(相当与odom)
class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        # 位置信息
        self.x = x
        self.y = y
        # 偏航角
        self.yaw = yaw
        # 速度
        self.v = v
        # 上一个时间步的转向角
        self.predelta = None

# 定义一个角度转换函数
def pi_2_pi(angle):
    return angle_mod(angle)

# 将车辆状态转为线性模型，并且写出状态矩阵A、输入矩阵B和状态偏移矩阵C
# 这里实际上是选用原点为参考点进行的线性化
def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C

# 画图函数，用于绘制车辆的位置和轨迹
def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

# 根据当前状态和控制输入(加速度和转向角)更新状态
# 实际上就是模型函数
def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state

# 将输入的矩阵（例如从优化求解器返回的矩阵）转换为一维的numpy数组。
def get_nparray_from_matrix(x):
    return np.array(x).flatten()

# 计算当前状态下离参考路径最近的点的索引。
# 输入当前车辆状态、参考路径的x坐标、y坐标、偏航角和起始索引(用于加快搜索速度)。
def calc_nearest_index(state, cx, cy, cyaw, pind):

    # 计算车辆从当前位置到参考路径上每个点的x与y坐标差值
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    # 计算车辆当前位置与参考路径点的平方距离列表
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    # 在平方距离列表中找到最小值
    mind = min(d)

    # 计算最小距离的索引并且加上起始索引
    ind = d.index(mind) + pind

    # 对最小平方距离去平方根,获得最小实际距离
    mind = math.sqrt(mind)

    # 计算车辆当前位置与参考路径点的x与y坐标差值
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    # 计算车辆当前位置与参考路径点的偏航角差值,并且归一化到-pi到pi之间
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    # 如果角度小于0,则将最小距离取负,表示车辆朝向与路进方向相反
    if angle < 0:
        mind *= -1

    # 返回最小距离的索引和最小距离
    return ind, mind

# 根据当前状态和控制输入预测未来一段时间内的车辆运动状态。
# x0：当前车辆状态 [x, y, v, yaw]。
# oa：最优控制序列中的,加速度输入序列。
# od：最优控制序列中的,转向角输入序列。
# xref：参考轨迹矩阵。
def predict_motion(x0, oa, od, xref):
    # 初始化xbar为与xref相同形状的零矩阵
    xbar = xref * 0.0

    # 将当前车辆状态赋值给xbar的第一列,作为初始状态
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    #使用state表示车辆状态,并且使其初始化为当前车辆状态
    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])

    # 循环T次,预测未来T个时间步内的车辆状态
    # 同时遍历加速度输入 oa、转向角输入 od 和时间步 i
    # ai：当前时间步的加速度输入
    # di：当前时间步的转向角输入
    # i：当前时间步的索引（从 1 到 T）
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        # 根据当前状态以及控制输入来更新车辆状态
        state = update_state(state, ai, di)
        # 将更新后的车辆状态存储到xbar中,并且与时间步i对应
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar

# 迭代线性MPC控制器
def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC control with updating operational point iteratively
    """
    """
    xref：参考轨迹矩阵，包含未来 T + 1 个时间步的参考状态
    x0：当前车辆状态 [x, y, v, yaw]
    dref：参考转向角序列
    oa：上一时间步的加速度输入序列(上一步MPC中计算得到的最优控制序列中的加速度输入序列)
    od：上一时间步的转向角输入序列(上一步MPC中计算得到的最优控制序列中的转向角输入序列)
    """

    # 初始化预测状态变量都为None
    ox, oy, oyaw, ov = None, None, None, None

    # 检查oa和od是否为None
    # 如果None，则初始化为长度为 T 的全零序列,说明是第一次迭代
    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    # 设置最多迭代次数为 MAX_ITER
    # 迭代的目的是逐步逼近最优控制序列。
    # 每次迭代过程中，
    # 使用上一次迭代的结果(最优控制序列以及使用最优控制序列得到的预测状态)
    # 作为当前的初始猜测，通过反复优化逐渐收敛到一个更优的解。
    for i in range(MAX_ITER):
        # 调用预测函数,根据当前状态与目标点状态以及之前的控制输入预测未来一段时间内的车辆运动状态
        xbar = predict_motion(x0, oa, od, xref)
        # 保存上一迭代步的控制输入
        poa, pod = oa[:], od[:]
        # 使用mpc计算新的最优控制序列以及预测状态序列
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        # 计算控制输入的变化值du
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        # 如果变化量小于阈值DU_TH,则跳出循环
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

# 线性MPC控制器
def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: 参考轨迹状态矩阵，包含未来 T + 1 个时间步的参考状态。
    xbar: 预测状态矩阵，包含未来 T + 1 个时间步的预测状态。
    x0: 初始状态 [x, y, v, yaw]
    dref: 参考转向角序列
    """

    # 优化变量矩阵，表示未来 T + 1 个时间步的状态
    x = cvxpy.Variable((NX, T + 1))
    # 优化变量矩阵，表示未来 T 个时间步的控制输入（加速度和转向角）
    u = cvxpy.Variable((NU, T))

    # 初始化成本函数为0
    cost = 0.0
    # 初始化约束条件列表
    constraints = []

    # 循环T次，构建未来T个时间步的成本函数和约束条件
    for t in range(T):

        # 最小化控制输入的成本,这里使用控制输入的二次型成本函数,权重矩阵为R
        cost += cvxpy.quad_form(u[:, t], R)

        # 如果不是第一个时间步
        if t != 0:
            # 最小化状态偏差(当前状态和参考状态的偏差)的二次型成本函数,权重矩阵为Q
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        # 获取线性模型的状态矩阵A、输入矩阵B和状态偏移矩阵C
        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        # 状态更新方程约束(x_t+1 = A*x_t + B*u_t + C)
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        # 如果在时间步t小于T-1,也就是不是最后一个时间步
        if t < (T - 1):
            # 最小化控制输入变化量(u_t+1 - u_t)的二次型成本函数,权重矩阵为Rd
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    # 最小化与最终状态的偏差(x_ref(T)-x(T))的二次型成本函数,权重矩阵为Qf
    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    # 设置初始状态约束,这一约束条件确保了MPC优化问题的初始状态与车辆的当前状态一致，从而保证预测的轨迹是从当前状态开始的
    constraints += [x[:, 0] == x0]

    # 添加速度的上下限约束
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]

    # 添加输入的(加速度和转向角)的上下限约束
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    # 使用cvxpy库构建二次型优化问题
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    # 使用cvxpy库的CLARABEL求解器(内点法)求解二次型优化问题
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    # 如果求解器的状态是OPTIMAL(最优)或者OPTIMAL_INACCURATE(近似最优),则将优化问题的解转换为numpy数组
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        # 提取优化问题的解
        # x_value是优化变量x的求解值,包含了最优预测轨迹的状态序列
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        # u.value是优化变量u的求解值,包含了要达到最优预测轨迹的最优控制输入序列
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    # 如果求解器的状态是INFEASIBLE(无解)或者其他状态,则打印错误信息,并且将优化问题的解设置为None
    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov

# 参考轨迹计算函数，用于计算给定状态下的参考轨迹
# 用于计算车辆当前状态下的参考轨迹
# 输入为车辆当前状态、路径的x坐标、y坐标、偏航角、曲率、目标速度列表、路径步长和当前目标点索引。
def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):

    # 初始化参考轨迹矩阵，维度为 NX（状态变量数量） 行 T + 1（预测时域长度）。
    xref = np.zeros((NX, T + 1))

    # 初始化偏航角差值矩阵，维度为 1 行 T + 1。
    dref = np.zeros((1, T + 1))

    # ncourse为路径点的数量
    ncourse = len(cx)

    # 计算车辆当前位置到参考路径的最近点的索引和距离
    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    # 如果计算得到的最近点索引大于等于当前目标点索引，则将当前目标点索引更新为最近点索引。
    # 为了确保车辆找到的是路径上最近的参考点
    if pind >= ind:
        ind = pind

    # 设置初始的参考点信息,即车辆当前应该跟随的路径点
    xref[0, 0] = cx[ind]  #目标x坐标
    xref[1, 0] = cy[ind]  #目标y坐标
    xref[2, 0] = sp[ind]  #目标速度
    xref[3, 0] = cyaw[ind] #目标偏航角
    dref[0, 0] = 0.0  # 设置为 0.0，表示初始的偏航角差值。

    # 用于记累计的路径长度
    travel = 0.0

    # 循环T+1次
    for i in range(T + 1):
        # 计算累计的距离,为速度*时间
        travel += abs(state.v) * DT

        # 参考点的索引偏移量(变化量),表示车辆在 i 个时间步后的目标点。
        dind = int(round(travel / dl))

        # 如果 ind + dind 小于路径点数量 ncourse，
        # 则设置 xref 和 dref 为 ind + dind 对应的路径点。
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        # 否则，设置 xref 和 dref 为路径的终点
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    # 返回参考轨迹矩阵,xref：参考轨迹矩阵，包含了未来 T + 1 个时间步的参考点。
    # ind：更新后的目标点索引。
    # dref：偏航角差值矩阵(始终为0)
    return xref, ind, dref

# 检查车辆是否到达目标点
def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False

# 仿真主函数
def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: 路径的x坐标列表
    cy: 路径的y坐标列表
    cy: 路径的偏航角列表
    ck: 路径的曲率列表
    sp: 目标速度列表
    dl: 路径步长,单位为m

    """

    # 读取路径中最后一个点的坐标作为最终目标点
    goal = [cx[-1], cy[-1]]

    # 设置初始状态
    state = initial_state

    # initial yaw compensation
    # 确保初始偏航角在-pi到pi之间
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0 # 初始化时间
    x = [state.x] # 记录车辆在每个时间步的 x 坐标。初始化时记录初始状态的 x 坐标
    y = [state.y] # 记录车辆在每个时间步的 y 坐标。初始化时记录初始状态的 y 坐标
    yaw = [state.yaw] # 记录车辆在每个时间步的偏航角。初始化时记录初始状态的偏航角
    v = [state.v] # 记录车辆在每个时间步的速度。初始化时记录初始状态的速度
    t = [0.0] # 记录每个时间步的时间。初始化时记录初始时间为0.0
    d = [0.0] # 记录每个时间步的转向角。初始化时记录初始状态的转向角为0.0
    a = [0.0] # 记录每个时间步的加速度。初始化时记录初始状态的加速度为0.0

    # 计算当前状态下车辆在路径上最近的目标点索引,与到索引的距离,起始索引为0
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    # 初始化用于存储上一时间步控制输入的变量
    # 在第一次计算控制输入时，这些变量为 None，
    # 后续迭代时将更新为前一时间步的控制输入值，以便进行预测和优化计算。
    # odelta：上一时间步的转向角列表。
    # oa：上一时间步的加速度列表。
    odelta, oa = None, None

    # 平滑路径偏航角序列中偏航角
    cyaw = smooth_yaw(cyaw)

    # 如果仿真时间小于最大仿真时间,则继续仿真
    while MAX_TIME >= time:

        # 计算当前车辆状态下的参考轨迹
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)
        
        # 将当前车辆状态存储到x0中
        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        # 使用迭代线性MPC控制器计算控制输入
        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)


        di, ai = 0.0, 0.0
        if odelta is not None:
            # 从MPC计算得到的最优控制序列中提取第一个控制输入作为当前时间步的控制输入。
            di, ai = odelta[0], oa[0]
            # 更新车辆状态
            state = update_state(state, ai, di)

        # 更新仿真时间
        time = time + DT

        # 记录更新后的车辆状态
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

        # 记录更新后的时间
        t.append(time)

        # 记录当前时间步的输入
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a

# 该函数计算沿参考路径的速度列表(输入为给定路径的位置与朝向信息,以及定义的每个路径点的目标速度)
# 本质是根据参考路径计算是使用倒车前进还是直接前进
def calc_speed_profile(cx, cy, cyaw, target_speed):

    # 创建一个长度与路径点相同的列表speed_profile
    # 初始值全部为目标速度
    speed_profile = [target_speed] * len(cx)

    # 设置方向为1.0,也就是前进方向
    direction = 1.0  # forward

    # Set stop point
    # 设置停止点并计算速度剖面
    # 遍历除最后一个点外的所有路径点
    for i in range(len(cx) - 1):

        # 计算相邻两点的x和y坐标差值
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        # 计算相邻点之间的方向(也就是从上一点到下一点需要将车身朝向哪个角度)
        # 从而计算每个点上的速度方向
        move_direction = math.atan2(dy, dx)

        # 如果dx和dy都不为0,证明车辆不是在直线上行驶
        if dx != 0.0 and dy != 0.0:
            # 计算移动方向与当前轨迹中偏航角的差值
            # 并且将角度归一化到-pi到pi之间并取绝对值
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            # 如果方向变化超过45度,则说明是很急的弯道,所以直接使用倒车行进
            if dangle >= math.pi / 4.0:
                if is_back:
                    direction = -1.0
                else:
                    direction = 1.0
            else:
                direction = 1.0

        # 如果方向不是1.0,则将速度列表中的值设置为负目标速度
        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    # 设置最后一个点的目标速度为0
    speed_profile[-1] = 0.0

    # 返回每个点的新目标速度列表
    return speed_profile

# 平滑路径上的偏航角序列，确保角度变化连续。
def smooth_yaw(yaw):

    # 遍历所有的偏航角列表中的每个元素,除了最后一个
    for i in range(len(yaw) - 1):
        # 计算相邻两个偏航角的差值
        dyaw = yaw[i + 1] - yaw[i]

        # 如果差值大于pi/2,则将第二个偏航角减去2*pi
        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            # 更新差值后再次计算
            dyaw = yaw[i + 1] - yaw[i]

        # 如果差值小于-pi/2,则将第二个偏航角加上2*pi
        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            # 更新差值后再次计算
            dyaw = yaw[i + 1] - yaw[i]

    # 返回平滑后的偏航角列表
    return yaw

# 生成一条直线路径。
def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

# 生成另一条稍微复杂的路径，包括直线和一些小的偏移。
def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

# 生成包含更大偏航角变化的路径。
def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck

# 生成一条更复杂的前进路径，包括多个转弯和直线段。
def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

# 生成一个包含多次急转弯和方向切换的复杂路径
def get_switch_back_course(dl):

    # 定义第一条需要经过的路径点
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]

    # 计算生成第一条三次样条曲线, 并且返回路径点的坐标、偏航角和曲率以及弧长(没有用到)
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    
    # 定义第二条需要经过的路径点
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]

    # 计算生成第二条三次样条曲线, 并且返回路径点的坐标、偏航角和曲率以及弧长(没有用到)
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    
    if is_back:
        # 将第二段路径的偏航角调整为与第一段路径方向相反。(为了模拟车辆的掉头行为)
        cyaw2 = [i - math.pi for i in cyaw2]


    # 将第二段路径的坐标、偏航角和曲率添加到第一段路径的坐标、偏航角和曲率中。
    # 也就是将两段路径拼接在一起。
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck

# 模拟了车辆在一个包含多次急转弯和方向切换的复杂路径上的行驶过程。路径使用了 get_switch_back_course 生成的复杂路径。
def main():
    # 打印当前文件路径
    print(__file__ + " start!!")
    # 记录开始时间
    start = time.time()

    # 设置路径步长
    dl = 1.0  # course tick
    # 生成不同的路径（这里选择了“switch-back”路径）
    # cx, cy, cyaw, ck = get_straight_course(dl)
    # cx, cy, cyaw, ck = get_straight_course2(dl)
    # cx, cy, cyaw, ck = get_straight_course3(dl)
    # cx, cy, cyaw, ck = get_forward_course(dl)
    cx, cy, cyaw, ck = get_switch_back_course(dl)

    # 计算速度剖面
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    # 初始化车辆状态
    # 输入路径第一个点的信息作为初状态
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    # 执行仿真
    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    # 计算并且打印仿真时间
    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    # 可视化
    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()

# 模拟了车辆在一个较为简单但包含大偏航角变化的路径上的行驶过程。路径使用了 get_straight_course3 生成的路径。
def main2():
    print(__file__ + " start!!")
    start = time.time()

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
    # main2()