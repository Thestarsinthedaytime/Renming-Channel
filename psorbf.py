# libraries
import random
import math
import scipy
from scipy import linalg
import numpy as np
import pandas as pd
from scipy.io import savemat

# 数据归一化 通用（0，1）归一化
def Normalization(data):
    for i in range(len(data[0])):
        # mencari nilai maksimum dan minimum setiap fitur
        max = np.amax(data[:, i])
        min = np.amin(data[:, i])

        # normalisasi menggunakan min-max normalization
        for j in range(len(data)):
            data[j][i] = (data[j][i] - min) / (max - min)

    return data

# data
def Data(file_name):
    # 读取 execl 表
    # file_name = 'datasets.xlsx'
    # data_training = pd.read_excel(file_name, sheet_name='PC5000_USE', usecols=range(0, 3))
    # train_target = pd.read_excel(file_name, sheet_name='PC5000_USE', usecols=[4])

    data_training = pd.read_excel(file_name, sheet_name='UseRBF_11', usecols=range(0, 3))
    train_target = pd.read_excel(file_name, sheet_name='UseRBF_11', usecols=[3])

    # 数据归一化
    Data.dt_training = Normalization(data_training.to_numpy())
    Data.dt_target_training = Normalization(train_target.to_numpy())

    return Data

def Hidden_layer(input_weights, biases, n_hidden_node, data_input):
    # W初始化  行数 = 隐藏层神经元个数   列数 = 输入数据的维度
    input_weight = input_weights.reshape(n_hidden_node, 3)
    # B初始化
    bias = biases.reshape(1, n_hidden_node)
    # # W的转置  行数 = 输入数据的维度   列数 = 隐藏层神经元个数
    # transpose_input_weight = np.transpose(input_weight)

    X = data_input
    c = input_weight
    delta = bias

    y = list()
    for i in range(X.shape[0]):
        x = X[i]
        x = np.reshape(x, (1, x.shape[0]))
        use_x = np.tile(x, (c.shape[0], 1))

        dist1 = (use_x - c) ** 2
        dist2 = dist1.sum(axis=1)
        dist3 = np.reshape(dist2, (1, dist2.shape[0]))
        delta2 = 2 * (delta ** 2)

        dist4 = dist3 / delta2
        rbf = np.exp( - dist4).reshape(c.shape[0])

        y.append(rbf)

    hidden_layer = np.array(y)

    return hidden_layer

# 计算输入矩阵的广义逆矩阵  matriks moore penrose pseudo-inverse menggunakan SVD
def Pseudoinverse(hidden_layer):
    h_pseudo_inverse = scipy.linalg.pinv2(hidden_layer, cond=None, rcond=None,
                                          return_rank=False, check_finite=True)
    return h_pseudo_inverse

# 求解伪逆矩阵，计算输出权重矩阵
def Output_weight(pseudo_inverse, target):
    beta = np.matmul(pseudo_inverse, target)

    return beta

# 计算得到神经网络预测值
def Target_output(testing_hidden_layer, output_weight):
    prediction = np.matmul(testing_hidden_layer, output_weight)# 矩阵乘法

    return prediction

# 产生一个粒子
def Particle(n_inputWeights, n_biases,population):
    # 每个粒子拥有多少个属性
    col = n_inputWeights + n_biases
    # 一共有多少个粒子
    row = population

    Particles = (np.random.rand(row, col) - 0.5) * 20

    # 输出是两个列表相加的新列表
    return Particles

# 所有粒子的初速度为零
def Velocity_0(particles):
    # 产生和粒子群维度相同的速度矩阵
    Velocity = np.zeros((particles.shape[0], particles.shape[1]))

    return Velocity

# 评价：正确率计算函数
def Evaluate(actual, prediction):
    # 进来两个（num,1）的数组
    cost = (actual- prediction)**2
    # mse = cost.mean()
    mse = cost.sum()
    
    return mse

# 将粒子属性及其适应度拼接起来
def Pbest(particles, fitness):
    fitness = np.expand_dims(fitness, axis=1)# 在fitness中间增加一个维度 第二维只有一个数 0
    pbest = np.hstack((particles, fitness))# 在水平方向上拼接  列拼接  增加列数  相当于每次的适应度都加上

    return pbest

# 将适应度最高的粒子挑选出来  全局最优值
def Gbest(particles, fitness):
    # # 获得最高的适应度
    # best_fitness = np.amax(fitness)

    # 获得最高适应度的粒子属性
    # particle = fitness.index(best_fitness)
    idx = np.argmax(fitness, axis=0)

    one = idx[0]
    best_fitness = fitness[one].reshape(1,1)
    best_particle = particles[one].reshape(1,particles.shape[1])

    # gbest
    # gbest = np.hstack((best_particle, best_fitness))
    gbest = [best_particle, best_fitness]

    return gbest

# 两代适应度比较更新
def Comparison(pbest_last, pbest_now):

    particles_last = pbest_last[0]
    fitness_last = pbest_last[1]

    particles_now = pbest_now[0]
    fitness_now = pbest_now[1]

    # 对每个粒子进行操作
    for i in range(min(len(particles_last), len(particles_now))):
        #比较适应度
        if fitness_last[i] > fitness_now[i]:
            particles_now[i] = particles_last[i]
            fitness_now[i] = fitness_last[i]
        # else:
        #     fitness_now[i] = pbest_t_1[i]

    return [particles_now, fitness_now]

# 粒子运动速度的更新
def Velocity_update(pbest, gbest, w, c1, c2, particles, velocity):
    # 遍历一个粒子的所有属性
    interval = []
    # 确定粒子每个属性  变化速度的最大最小幅度
    for j in range(len(particles[0])):
        x_max = np.amax(np.array(particles)[:, j])
        x_min = np.amin(np.array(particles)[:, j])
        k = random.random()       # 保留一位随机小数

        v_max_j = np.array(((x_max - x_min) / 2) * k)
        v_min_j = np.array(v_max_j * -1)
        # 在水平方向上平铺
        intvl = np.hstack((v_min_j, v_max_j))
        interval.append(intvl)
    interval = np.transpose( np.array(interval) )

    # 每个属性变化速度的更新  [0,1)
    r1 = random.random()
    r2 = random.random()

    # 对种群中每个粒子进行操作    , len(gbest), len(pbest)
    p_particle = pbest[0]
    g_particle = gbest[0]
    # velocity_now = np.zeros(velocity.shape)
    velocity_now = np.random.rand(velocity.shape[0], velocity.shape[1])

    for i in range(min(len(particles), len(velocity))):
        for j in range(len(particles[0])):
            # print(i,j)
            a = (w * velocity[i][j])
            b = (c1 * r1 * (p_particle[i][j] - particles[i][j]))
            c = (c2 * r2 * (g_particle[0][j] - particles[i][j]))

            the_v =  a + b + c

            if the_v > interval[1][j]:
                the_v = interval[1][j]
            elif the_v < interval[0][j]:
                the_v = interval[0][j]

            velocity_now[i][j] = the_v

    return velocity_now

# 粒子位置的更新
def Position_update(position_last, velocity_new):
    return position_last + velocity_new


# fungsi ELM
def Elm(Data, particles, n_input_weights, n_hidden_node):
    # 定义一个适应度列表
    fitness = []
    # output_training =
    for i in range(len(particles)):
        # -----------------training---------------------#
        # 参数读取 都是 (n,) 的 np数组
        input_weights = particles[i][0:n_input_weights]
        biases = particles[i][n_input_weights:len(particles[i])]

        # 计算隐藏层输出的结果：  Activate ( WX + B ) 矩阵
        hidden_layer_training = Hidden_layer(input_weights, biases, n_hidden_node, Data.dt_training)

        # 计算得到输入矩阵的伪逆矩阵
        pseudo_training = Pseudoinverse(hidden_layer_training)

        # 伪逆矩阵和目标输出相乘，求解输出权重矩阵  就是目标
        output_training = Output_weight(pseudo_training, Data.dt_target_training)

        # -----------------testing--------------------#
        # # 计算得到测试数据集输出
        # hidden_layer_testing = Hidden_layer(input_weights, biases, n_hidden_node, Data.dt_testing)
        # prediction = Target_output(hidden_layer_testing, output_training)

        # 计算得到训练集预测输出 激活函数输出结果 * 输出层权重
        prediction = Target_output(hidden_layer_training, output_training)

        # 计算训练集的适应度
        cost = Evaluate(Data.dt_target_training, prediction)
        fitness.append(1/cost)

    fitness = np.array(fitness).reshape(len(particles),1)

    return fitness

# 参数提取和重新存储
def save_result(Data, best_one, n_input_weights, n_hidden_node):
    input_weights = best_one[0:n_input_weights]
    biases = best_one[n_input_weights:len(best_one)]
    # W初始化  行数 = 隐藏层神经元个数   列数 = 输入数据的维度
    input_weight = input_weights.reshape(n_hidden_node, 3)
    # B初始化
    bias = biases.reshape(1, n_hidden_node)
    # # W的转置  行数 = 输入数据的维度   列数 = 隐藏层神经元个数
    # transpose_input_weight = np.transpose(input_weight)

    X = Data.dt_training

    c = input_weight
    delta = bias

    y = list()
    for i in range(X.shape[0]):
        x = X[i]
        x = np.reshape(x, (1, x.shape[0]))
        use_x = np.tile(x, (c.shape[0], 1))

        dist1 = (use_x - c) ** 2
        dist2 = dist1.sum(axis=1)
        dist3 = np.reshape(dist2, (1, dist2.shape[0]))
        delta2 = 2 * (delta ** 2)

        dist4 = dist3 / delta2
        rbf = np.exp(- dist4).reshape(c.shape[0])

        y.append(rbf)

    O1 = np.mat(y)

    pseudo_inverse = scipy.linalg.pinv2(O1, cond=None, rcond=None, return_rank=False, check_finite=True)

    beta = np.matmul(pseudo_inverse, Data.dt_target_training)

    file_name = 'D:\pycharm_project\RBF_RenMing1\Model_PSO\RBF_data.mat'
    savemat(file_name, {'c': c, 'delta': delta, 'beta': beta})
