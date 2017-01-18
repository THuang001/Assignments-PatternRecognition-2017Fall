## Ng spectral clustering

import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

with open('X_spectral.txt', encoding='utf-8') as fr:
    X = [(float(line.strip().split()[0]), float(line.strip().split()[1])) for line in fr.readlines()]
X = np.array(X)

def visual(X, k=None, sigma=None, save=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:100, :1], X[:100, 1:], c='blue')
    ax.scatter(X[100:, :1], X[100:, 1:], c='red')
    if save != 0:
        plt.title('data after transforming\n($\sigma=' + str(sigma) + '$, $k=' + str(k) + '$)')
        plt.savefig('k' + str(k) + '_' + 'sigma' + str(sigma)  + '.pdf', dpi=400)
    plt.show()

def get_L(X, k, sigma):
    """
    生成对称型拉普拉斯矩阵L_sym
    :param X: 数据点
    :param k: 参数
    :param sigma: 参数
    :return: L_sym
    """
    (n, d) = X.shape
    D = np.zeros((n, n))  # 度矩阵
    W = np.zeros((n, n))  # 点对亲和性矩阵
    for i in range(n):
        Xi_neibors = [np.linalg.norm(X[i] - X[j]) for j in range(n)]  # k近邻生成边
        neibors_index = np.argsort(Xi_neibors)[1:(k+1)]
        for index in neibors_index:
            W[i][index] = np.exp(-(np.linalg.norm(X[i] - X[index]))**2 / (2 * sigma**2))
    W = (W + np.transpose(W)) / 2  # 保证其是对称矩阵, 修正k近邻生成的图
    for i in range(n):
        D[i][i] = sum(W[i])
    L = D - W
    D_ = np.zeros((n, n))
    for i in range(n):
        D_[i][i] = pow(D[i][i], -0.5)
    L_sym = np.dot(np.dot(D_, L), D_)
    return L_sym, L

def spectral(X, sigma, k, centroids):
    """
    Ng谱聚类算法
    :param X: 数据点
    :param sigma: 参数
    :param k: 参数
    :return: accu聚类精度
    """
    (n, d) = X.shape
    L_sym, L = get_L(X, k, sigma)
    eig, eigvec = np.linalg.eig(L_sym)  # eigvec按列
    # eig_index = np.argsort(eig)[1:d+1]
    eig_index = np.argsort(eig)[:d] # 最小的d个特征值的索引
    U = eigvec[:, eig_index]
    T = np.zeros(U.shape)
    for i in range(n):
        for j in range(d):
            T[i][j] = U[i][j] / np.linalg.norm(U[i])
    Y = T
    # visual(Y, k=k, sigma=sigma, save=1)

    cluster = KMeans(2, 100, centroids)
    cluster.fit(Y)
    labels = cluster.labels

    if labels[0] == 0:
        n1 = 100 - sum(labels[:100]); n2 = sum(labels[100:])
    else:
        n1 = sum(labels[:100]); n2 = 100 - sum(labels[100:])
    accu = (n1 + n2) / n
    print('---------------------sigma=%.2f, k=%d, accu=%.4f' % (sigma, k, accu))
    return accu

def visualExperiment(para_list, accu_list, C, paraISk=1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(para_list, accu_list, marker='s', c='blue', lw=1.5)
    plt.ylim(0.6, 1.02)
    plt.ylabel('$accu$')
    if paraISk == 1:
        plt.xlabel('$k$')
        plt.title('$accu$ with different $k$\n($\sigma=0.5$)')
        # plt.savefig('spectral_k.pdf', dpi=400)
    else:
        plt.xlabel('$\sigma$')
        plt.title('$accu$ with different $\sigma$\n($k=10$)')
        # plt.savefig('spectral_sigma.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    centroids = None
    # visual(X)
    sigma = 0.5
    k_list = range(10, 60, 5)
    accu_list_k = []
    for k in k_list:
        accu_list_k.append(spectral(X, sigma, k, centroids))
    visualExperiment(k_list, accu_list_k, sigma, paraISk=1)

    k = 20
    sigma_list = range(2, 22, 2); sigma_list = [_/10 for _ in sigma_list]  # 0.2, 0.4, 0.6, ..., 2.0
    accu_list_sigma = []
    for sigma in sigma_list:
        accu_list_sigma.append(spectral(X, sigma, k, centroids))
    visualExperiment(sigma_list, accu_list_sigma, k, paraISk=0)
