# libraries
import psorbf as pso

# training_dataset
file_name = 'E:/result/training_dataset.xlsx'

fitures = 3   #输入维度
n_hidden_node = 20  # 隐藏层神经元个数
n_input_weights = n_hidden_node * fitures  # 待优化的权重参数个数
population = 100  # 粒子种群规模
max_iter = 21  # 最大迭代次数
w = 0.5  # bobot inersia
c1 = 2  # kontansta kecepatan 1
c2 = 2  # konstanta kecepatan 2

# 训练数据集载入
Data = pso.Data(file_name)

# 粒子初始化： 是一个np二维数组  第一个索引为粒子号  第二个索引为参数号
particles_last = pso.Particle(n_input_weights, n_hidden_node, population)# 每个粒子200个参数

# # 每个粒子的初始速度都为零  与粒子的数据格式相同
velocity_last = pso.Velocity_0(particles_last)

# 种群的初始适应度：正确率  每个粒子的适应度
fitness_last = pso.Elm(Data, particles_last, n_input_weights, n_hidden_node)

# 将适应度作为粒子的一个属性
pbest_last = [particles_last, fitness_last]

# 挑选出粒子最高的适应度和属性
gbest_last = pso.Gbest(particles_last, fitness_last)
fitness_best_last = gbest_last[1][0][0]

a = 0
##############################################################################################
for iteration in range(max_iter):
    # 粒子速度更新
    velocity_t = pso.Velocity_update(pbest_last, gbest_last, w, c1, c2, particles_last, velocity_last)

    # 粒子位置更新
    particles_t = pso.Position_update(particles_last, velocity_t)

    # 极限学习机评价适应度
    fitness_t = pso.Elm(Data, particles_t, n_input_weights, n_hidden_node)

    # 得到每个粒子历史上的最优值
    pbest_t = [particles_t, fitness_t]
    pbest_t = pso.Comparison(pbest_last, pbest_t)

    # 找出当前粒子中的最优粒子
    gbest_t = pso.Gbest(particles_t, fitness_t)
    fitness_best_t = gbest_t[1][0][0]

    # -----------------------------参数动态调整------------------------------#
    if (fitness_best_t <= fitness_best_last):
        a = a + 1
        if a == 2:
            w = w * 0.8
            a = 0
    else:
        a = 0

    # -----------------------------迭代结果存储------------------------------#
    if (iteration == max_iter-1):
        best_one = gbest_t[0][0]
        pso.save_result(Data, best_one, n_input_weights, n_hidden_node)

    # -----------------------------残差信息显示------------------------------#
    Accuracy = (1 / fitness_best_t) / 3899
    print('')
    print('Accuracy ' + str(iteration))
    print(w, Accuracy)

    # -----------------------------变量更新换代------------------------------#
    pbest_last = pbest_t
    gbest_last = gbest_t
    particles_last = particles_t
    velocity_last = velocity_t
    fitness_best_last = fitness_best_t
