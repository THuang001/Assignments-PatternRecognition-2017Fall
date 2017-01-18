## BP for a three layers Neural Networks, mean square loss

import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def batchGD_bp(X, y, d=3, nH=10, c=3, lr=0.8, T=100, eps=0.0):
    """
    BP算法, 每轮迭代使用全部样本
    :param X: 训练样本的特征矩阵
    :param y: 训练样本的标签向量
    :param d: 训练样本的特征维数
    :param nH: 隐层的节点数
    :param c: 类别数
    :param lr: 学习率
    :param T: 停机条件1(最大迭代轮数)
    :param eps: 停机条件2(相邻两次迭代loss之差的最大允许值), 设为0.0表示不使用这个条件
    :return:
    """
    W_H = np.random.normal(size=(nH, d))  # np.random.random(size=(nH, d))  # [0.0, 1.0)之间均匀分布
    b_H = np.array([0.0] * nH).reshape(nH, 1)
    W_c = np.random.normal(size=(c, nH))
    b_c = np.array([0.0] * c).reshape(c, 1)
    Loss = []; loss = 0; false_num = []
    for t in range(T):
        loss_last = loss
        y_ = []
        for idx, x in enumerate(X):
            ## 前向传播
            x = x.reshape(d, 1)
            net_H = np.dot(W_H, x) + b_H
            z_H = np.tanh(net_H)
            net = np.dot(W_c, z_H) + b_c
            z = sigmoid(net)
            y_.append(z.argmax())
            y_x = y[idx].reshape(d, 1)
            loss = 0.5 * np.sum(np.square(y_x - z))
            ## 误差反向传播
            # 输出层
            delta_c = z * (1 - z) * (z - y_x)  # element-wise
            grad_Wc = np.dot(delta_c, np.transpose(z_H))
            grad_bc = delta_c
            W_c -= lr * grad_Wc
            b_c -= lr * grad_bc
            # 隐层
            delta_H = (1 - np.square(z_H)) * (np.dot(np.transpose(W_c), delta_c))
            grad_WH = np.dot(delta_H, np.transpose(x))
            grad_bH = delta_H
            W_H -= lr * grad_WH
            b_H -= lr * grad_bH
        Loss.append(loss)
        ## 计算本轮过后错分的样本数
        y_ = np.array(y_).reshape((30,))
        tOf = (np.argmax(y, axis=1) == y_)
        false_num.append(np.where(tOf == False)[0].shape[0])
        if false_num[-1] == 0:  # or abs(loss_last - loss) <= eps:  # 停机条件
            return t, Loss, false_num
    return T, Loss, false_num

def singleGD_bp(X, y, d=3, nH=10, c=3, lr=0.8, T=100, eps=0.0):
    """
    BP算法, 每轮迭代只用一个样本
    :param X: 训练样本的特征矩阵
    :param y: 训练样本的标签向量
    :param d: 训练样本的特征维数
    :param nH: 隐层的节点数
    :param c: 类别数
    :param lr: 学习率
    :param T: 停机条件1(最大迭代轮数)
    :param eps: 停机条件2(相邻两次迭代loss之差的最大允许值), 设为0.0表示不使用这个条件
    :return:
    """
    W_H = np.random.normal(size=(nH, d))  # np.random.random(size=(nH, d))  # [0.0, 1.0)之间均匀分布
    b_H = np.array([0.0] * nH).reshape(nH, 1)
    W_c = np.random.normal(size=(c, nH))
    b_c = np.array([0.0] * c).reshape(c, 1)
    Loss = []; loss = 0
    for t in range(T):
        loss_last = loss
        y_ = []
        ## 前向传播
        idx = np.random.choice(30, 1)
        x = X[idx].reshape(d, 1)
        net_H = np.dot(W_H, x) + b_H
        z_H = np.tanh(net_H)
        net = np.dot(W_c, z_H) + b_c
        z = sigmoid(net)
        y_.append(z.argmax())
        y_x = y[idx].reshape(d, 1)
        loss = 0.5 * np.sum(np.square(y_x - z))
        ## 误差反向传播
        # 输出层
        delta_c = z * (1 - z) * (z - y_x)  # element-wise
        grad_Wc = np.dot(delta_c, np.transpose(z_H))
        grad_bc = delta_c
        W_c -= lr * grad_Wc
        b_c -= lr * grad_bc
        # 隐层
        delta_H = (1 - np.square(z_H)) * (np.dot(np.transpose(W_c), delta_c))
        grad_WH = np.dot(delta_H, np.transpose(x))
        grad_bH = delta_H
        W_H -= lr * grad_WH
        b_H -= lr * grad_bH
        Loss.append(loss)
        if abs(loss_last - loss) <= eps:  # 停机条件
            return t, Loss
    return T, Loss

def visual_nH(nH_list):
    """
    可视化训练精度与隐层节点个数的关系. 这里的绘制方式是, 指定迭代轮数, 绘制每一个节点数目在迭代结束后的训练精度
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    Loss_list = []
    for i, nH in enumerate(nH_list):
        _, Loss, _ = batchGD_bp(X, y, nH=nH, T=500)
        Loss_list.append(Loss[-1])
    ax.plot(nH_list[0]+np.arange(len(nH_list)), Loss_list, marker ='s', c='red', lw=1.5)
    plt.xlabel('$n_H$'); plt.ylabel('$loss$')
    plt.title('$loss$ on training data with different $n_H$\n(after $iters=500$)')
    plt.savefig('nH.pdf', dpi=400)
    plt.show()

def visual_lr(lr_list, color_list):
    """
    可视化训练精度与学习率的关系. 这里的绘制方式是, 指定迭代轮数, 绘制每一个学习率在迭代过程中的训练精度曲线
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    Loss_list = []
    for i, lr in enumerate(lr_list):
        _, Loss, _ = batchGD_bp(X, y, lr=lr, T=25)
        Loss_list.append(Loss)
        ax.plot(np.arange(25), Loss_list[i], c=color_list[i], lw=1.5, label='$'+str(lr)+'$')  # marker='s'
    plt.xlabel('$step$'); plt.ylabel('$loss$ with different learning rate')
    plt.title('$loss$ on training data with different learning rate')
    plt.legend(loc=1)
    plt.savefig('lr' + time.strftime('%m%d%H%M%S') + '.pdf', dpi=400)
    plt.show()

def visual_loss(iters, Loss):
    """
    可视化损失函数随迭代步数变化的曲线
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(iters+1), Loss, c='purple', lw=5)
    plt.xlabel('$step$'); plt.ylabel('$loss$')
    plt.title('$loss$ on training data')
    plt.savefig('loss' + time.strftime('%m%d%H%M%S') + '.pdf', dpi=400)
    plt.show()

if __name__ == '__main__':
    X = np.array([[ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63], [-1.49, 2.18, -3.39],
                  [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87], [ 1.20, 1.40, -1.89], [ -0.92, 1.44, -3.22],
                  [ 0.45, 1.33, -4.38], [-0.76, 0.84, -1.96], [ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8],
                  [ 0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
                  [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14], [ 0.46, 1.49, 0.68],
                  [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69], [1.86, 3.19, 1.51],
                  [1.68, 1.79, -0.87], [3.51, -0.22, -1.39], [1.40, -0.44, -0.92], [0.44, 0.83, 1.97],
                  [0.25, 0.68, -0.99], [ 0.66, -0.45, 0.08]])

    y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    # 第一问
    visual_nH(list(range(3, 15)))
    # 第二问
    visual_lr([0.3, 0.6, 0.9, 1.2, 1.5], ['red', 'blue', 'yellow', 'green', 'purple'])
    # 第三问
    iter_batch, Loss_batch, false_num_batch = batchGD_bp(X, y, T=1000)
    visual_loss(iter_batch, Loss_batch)
