## Batch Perception、Ho-Kashyap、Batch Relaxation with Margin、Single-sample Relaxation with Margin

import numpy as np
import matplotlib.pyplot as plt
import time

Y1 = np.array([[1, 0.1, 1.1], [1, 6.8, 7.1], [1, -3.5, -4.1], [1, 2.0, 2.7], [1, 4.1, 2.8], [1, 3.1, 5.0], [1, -0.8, -1.3], [1, 0.9, 1.2], [1, 5.0, 6.4], [1, 3.9, 4.0]])
Y2 = np.array([[1, 7.1, 4.2], [1, -1.4, -4.3], [1, 4.5, 0.0], [1, 6.3, 1.6], [1, 4.2, 1.9], [1, 1.4, -3.2], [1, 2.4, -4.0], [1, 2.5, -6.1], [1, 8.4, 3.7], [1, 4.1, -2.2]])
Y3 = np.array([[1, -3.0, -2.9], [1, 0.5, 8.7], [1, 2.9, 2.1], [1, -0.1, 5.2], [1, -4.0, 2.2], [1, -1.3, 3.7], [1, -3.4, 6.2], [1, -4.1, 3.4], [1, -5.1, 1.6], [1, 1.9, 5.1]])
Y4 = np.array([[1, -2.0, -8.4], [1, -8.9, 0.2], [1, -4.2, -7.7], [1, -8.5, -3.2], [1, -6.7, -4.0], [1, -0.5, -9.2], [1, -5.3, -6.7], [1, -8.7, -6.4], [1, -7.1, -9.7], [1, -8, -6.3]])

N = len(Y1)  # 10
d = len(Y1[0])  # 3

def visualDicisionBoundry(Ym, Yn, m, n, a, title):
    """
    绘制两类样本点、决策边界
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    # 数据点
    ax.scatter(Ym[:, 1:2], Ym[:, 2:], s=30, label='$\omega_' + str(m) + '$')
    ax.scatter(Yn[:, 1:2], Yn[:, 2:], s=30, label='$\omega_' + str(n) + '$', color='r')
    # 决策边界: a.T * (1, x1, x2) = 0
    x1 = np.arange(-6, 7.5, 0.1)
    ax.plot(x1, (- a[1] * x1 - a[0]) / a[2], color='black', lw=1)
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend(loc=2)
    # plt.savefig(title + time.strftime('%m%d%H%M%S') + '.pdf', dpi=400)
    plt.show()


def visualCriterion(k1, k2, criterion1, criterion2, title):
    """
    绘制准则函数随着迭代次数变化的曲线
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, k1, 1), criterion1, marker ='s', c='red', lw=1.5, label='$b=0.1$')
    ax.plot(np.arange(0, k2, 1), criterion2, marker ='s', c='blue', lw=1.5, label='$b=0.5$')
    plt.xlabel('$k$'); plt.ylabel(r'$J_r$')
    plt.title(title)
    plt.legend(loc=1)
    # plt.savefig(title + time.strftime('%m%d%H%M%S') + '.pdf', dpi=400)
    plt.show()


def batchPerception(a1, eta1, Ym, Yn, m, n):
    """
    批处理感知器算法
    """
    Y = np.append(Ym, -Yn, axis=0)  # 负类样本规范化
    a = a1; eta = eta1
    step = 0
    for k in range(2, 1000):
        yk = [i.reshape(d, 1) for i in Y if np.dot(np.transpose(a), i.reshape(d, 1)) <= 0]  # print(k-2, len(Yk))
        if not yk:
            visualDicisionBoundry(Ym, Yn, m, n, a, 'Batch Perception')
            return a, k-2
        a = a + eta * sum(yk)
        step = k-1
    visualDicisionBoundry(Ym, Yn, m, n, a, 'Batch Perception')
    return a, step


def hoKashyap(b1, eta1, b_min, Ym, Yn, m, n):
    """
    Ho-Kashyap算法
    """
    Y = np.append(Ym, -Yn, axis=0)  # 负类样本需要规范化
    b = b1
    a = np.dot(np.linalg.pinv(Y), b); eta = eta1
    for k in range(2, 1000):
        e = np.dot(Y, a) - b
        ee = 0.5 * (e + abs(e))
        b = b + 2 * eta * ee
        a = np.dot(np.linalg.pinv(Y), b)
        if len(e[np.where(abs(e) > b_min)]) == 0:  # 该条件满足则算法收敛
            visualDicisionBoundry(Ym, Yn, m, n, a, 'Ho-Kashyap')
            return np.linalg.norm(e)**2, k-1
        if len(e[np.where(e <= 0)]) == 2*N:  # 该条件满足则说明样本线性不可分: 误差向量e是没有正分量的非零向量
            visualDicisionBoundry(Ym, Yn, m, n, a, 'Ho-Kashyap')
            return np.linalg.norm(e)**2, k-1


def batchRelaxationMargin(a1, eta1, b, Ym, Yn, m, n):
    """
    带裕量的批处理松弛算法
    """
    Y = np.append(Ym, -Yn, axis=0)  # 负类样本规范化 shape=(20, 3)
    a = a1; eta = eta1
    criterion = []
    step = 0
    while True:
        Y_error = [x for x in Y if np.dot(x, a) <= b]  # 当前迭代时分错的样本集合
        print(step, len(Y_error))
        if not len(Y_error):  # 收敛
            visualDicisionBoundry(Ym, Yn, m, n, a, 'Batch Relaxation with Margin($\eta_k=' + str(eta1) + ')$')
            return step, criterion

        criterion.append(np.sum([0.5 * (np.dot(y, a) - b)**2 / (np.linalg.norm(y))**2 for y in Y_error]))
        a = a + eta * sum([(b - (np.dot(y, a)) / (np.linalg.norm(y))**2) * y.reshape(d, 1) for y in Y_error])
        # for y in Y_error: a = a + eta * (b - (np.dot(y, a)) / (np.linalg.norm(y))**2) * y.reshape(d, 1)

        if step > 1000:  # 不收敛
            visualDicisionBoundry(Ym, Yn, m, n, a, 'Batch Relaxation with Margin($\eta_k=' + str(eta1) + ')$')
            return step+1, criterion
        step += 1


def singleRelaxationMargin(a1, eta1, b, Ym, Yn, m, n):
    """
    带裕量的单样本松弛算法
    """
    Y = np.append(Ym, -Yn, axis=0)  # 负类样本规范化 shape=(20, 3)
    # 为使样本近似无穷次出现, 不断将两类样本加到Y中
    for i in range(100):
        if i % 2 == 0: Y = np.append(Y, Ym, axis=0)
        else: Y = np.append(Y, -Yn, axis=0)
    a = a1; eta = eta1
    num = 0  # 记录连续多少个样本没被错误分类
    cri = []
    step = 0
    for k in range(2, len(Y)):
        yk = Y[k-2]
        if np.dot(yk, a) <= b:  # 更新参数
            a = a + eta * ((b - np.dot(yk, a))/((np.linalg.norm(yk))**2) * yk.reshape(d, 1))
            num = 0
            Y_error = [y for y in Y if np.dot(y, a) <= b]  # 当前迭代时分错的样本集合
            cri.append(np.sum([0.5 * (np.dot(y, a) - b)**2 / (np.linalg.norm(y))**2 for y in Y_error]))  # 当前迭代时的准则函数值
            step += 1
        else:
            num += 1
        if num == 2*N:  # 连续20个样本都没被错误分类, 算法收敛
            visualDicisionBoundry(Ym, Yn, m, n, a, 'Single-sample Relaxation with Margin($\eta_k=' + str(eta1) + ')$')
            return step+1, cri
    # 不收敛
    visualDicisionBoundry(Ym, Yn, m, n, a, 'Single-sample Relaxation with Margin($\eta_k=' + str(eta1) + ')$')
    return step, cri


if __name__ == '__main__':
    ## 第一题, Batch Perception
    a1 = np.zeros(shape=(d, 1), dtype=float)
    eta1 = 1
    a12, step12 = batchPerception(a1, eta1, Y1, Y2, 1, 2); print(a12, step12)
    a32, step32 = batchPerception(a1, eta1, Y3, Y2, 3, 2); print(a32, step32)

    ## 第二题, Ho-Kashyap
    b1 = np.array([[1] * (2*N)]).reshape((2*N), 1)
    eta1 = 0.8
    b_min = 0.1
    J13, step13 = hoKashyap(b1, eta1, b_min, Y1, Y3, 1, 3); print(J13, step13)
    J24, step24 = hoKashyap(b1, eta1, b_min, Y2, Y4, 2, 4); print(J24, step24)

    ## 第三题, Batch Relaxation with Margin
    a1 = np.zeros(shape=(d, 1), dtype=float)
    eta1 = 1
    step1, cri1 = batchRelaxationMargin(a1, eta1, 0.1, Y2, Y3, 2, 3); print(step1, '\n', cri1)
    step2, cri2 = batchRelaxationMargin(a1, eta1, 0.5, Y2, Y3, 2, 3); print(step2, '\n', cri2)
    visualCriterion(step1, step2, cri1, cri2, 'Batch Relaxation with Margin: Criterion($\eta_k=' + str(eta1) + ')$')
    ## 第三题, Single-sample Relaxation with Margin
    step1, cri1= singleRelaxationMargin(a1, eta1, 0.1, Y2, Y3, 2, 3); print(step1, '\n', cri1)
    step2, cri2= singleRelaxationMargin(a1, eta1, 0.5, Y2, Y3, 2, 3); print(step2, '\n', cri2)
    visualCriterion(step1, step2, cri1, cri2, 'Single-sample Relaxation with Margin: Criterion\n($\eta_k=' + str(eta1) + ')$')

