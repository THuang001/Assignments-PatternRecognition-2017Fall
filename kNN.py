## kNN

import numpy as np
from load_MNIST import load_X, load_y
from collections import Counter
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os

def inputData():
    """
    载入训练集和测试集
    :return: ndarray
        train_X: 全部训练样本的特征矢量; train_y: 全部训练样本的标签
        test_X: 全部测试样本的特征矢量; test_y: 全部测试样本的标签
    """
    trainfile_X = 'MNIST/train-images-idx3-ubyte'
    trainfile_y = 'MNIST/train-labels-idx1-ubyte'
    testfile_X = 'MNIST/t10k-images-idx3-ubyte'
    testfile_y = 'MNIST/t10k-labels-idx1-ubyte'
    if not os.path.exists('train_X.npy'):  # 避免重复载入, 如果已存在就跳过这步
        load_X(trainfile_X, 'train_X.npy')
        load_y(trainfile_y, 'train_y.npy')
        load_X(testfile_X, 'test_X.npy')
        load_y(testfile_y, 'test_y.npy')
    return np.load('train_X.npy'), np.ravel(np.load('train_y.npy')), \
           np.load('test_X.npy'), np.ravel(np.load('test_y.npy'))

def my_kNN(train_X, train_y, test_X, test_y, k):
    """
    kNN分类
    :param train_X: 全部训练样本的特征矢量
    :param train_y: 全部训练样本的标签
    :param test_X: 全部测试样本的特征矢量
    :param test_y: 全部测试样本的标签
    :param k: 近邻的个数, 升序排列的列表
    :return: 模型在测试集上的accuracy
    """
    accuracy = [0] * len(k)  # 存放不同k值下的accuracy
    n_testsamples = test_X.shape[0]
    for i in range(n_testsamples):
        Xi = test_X[i]
        # dist = [np.linalg.norm(Xi - j) for j in train_X]  # Xi与全部训练样本的欧氏距离
        dist = [cosine(Xi, j) for j in train_X]  # Xi与全部训练样本的cos相似度(返回的是1-cos, 所以值越小代表越相似)
        candicate = np.argsort(dist)[0:max(k)]  # 升序排序后的索引数组
        for j in range(len(k)):  # 不同的k值依次计算accuracy
            yi = Counter(train_y[candicate[0:k[j]]]).most_common(1)[0][0]
            if yi == test_y[i]:
                accuracy[j] += 1
    accuracy = np.array(accuracy) / n_testsamples
    return accuracy

def experimentsVisualize(para, accuracy):
    """
    不同的k值下, 测试集上的准确率
    :param para: 不同的k值
    :param accuracy: 不同的k值下, 测试集上的准确率
    :return: 无
    """
    accuracy = 100 * np.array(accuracy)
    fig1 = plt.figure()  # figsize=(10, 8)
    ax1 = fig1.add_subplot(111)
    ax1.plot(para, accuracy, marker = 's', c='red', lw=1.5, label='$Accuracy$')
    for i in range(len(para)):
        plt.text(para[i]-1.2, accuracy[i]+1, str(round(accuracy[i],2)))
    plt.legend(loc=4)
    plt.xlabel('$k$')
    plt.ylabel('$Accuracy(\%)$')
    plt.title('$Accuracy(\%)$ by $k$NN')
    plt.ylim(75, 100)
    plt.xlim(-2, max(para)+3)
    plt.savefig('5_cos.pdf', dpi=400)
    # plt.show()

if __name__=='__main__':

    train_X, train_y, test_X, test_y = inputData()
    para = [1, 5, 15, 25, 35]
    accuracy = my_kNN(train_X, train_y, test_X, test_y, para)
    for i in range(len(para)):
        print('kNN(mine):    k=%d,   accuracy = %.2f%%' % (para[i], 100 * accuracy[i]))

    experimentsVisualize(para, accuracy)