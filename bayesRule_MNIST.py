## Bayes Rule：Linear Discriminant Function、Quadratic Discriminant Function(with Shrinkage)

import numpy as np
from load_MNIST import load_X, load_y
from sklearn.decomposition import PCA
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

def pcaFeatures(train_X, test_X, k):
    """
    特征降维
    :param train_X: 全部训练样本的特征矢量
    :param test_X: 全部测试样本的特征矢量
    :param k: 指定降维之后的特征维数
    :return: 降维后训练样本和测试样本的特征矢量
    """
    pca = PCA(n_components=k)
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)
    print(sum(pca.explained_variance_ratio_))
    return train_X, test_X

def linearDF(train_X, train_y, test_X, test_y, beta=0.0, prior=None):
    """
    线性判别函数LDF, 包括训练和测试
    :param train_X: 全部训练样本的特征矢量
    :param train_y: 全部训练样本的标签
    :param test_X: 全部测试样本的特征矢量
    :param test_y: 全部测试样本的标签
    :param beta: Shrinkage(将协方差矩阵向单位矩阵缩并)的参数值
    :param prior: 各个类的先验概率, 如不指定将自行计算
    :return: 模型在测试集上的accuracy
    """
    n_samples, n_features = train_X.shape
    ## 计算各个类别的先验概率
    classLabel, y = np.unique(train_y, return_inverse=True)
    if prior is None:
        prior = np.bincount(y) / float(n_samples)
    C = len(classLabel)

    ## I. 训练, 使用MLE来估计各个类别的均值向量
    Mu = np.zeros(shape=(C, n_features))
    for c in classLabel:
        train_Xc = train_X[y==c, :]
        Mu[c] = np.mean(train_Xc, axis=0)
    # 各个类别的协方差矩阵相同, 需要在归一化方差为1之后求得, 并求协方差矩阵的逆矩阵
    Var = np.var(train_X, axis=0)
    for i in range(n_features):
        if Var[i] == 0:
            Var[i] = 1.0e-4
    Sigma = np.cov(train_X / Var, rowvar=False)
    # Shrinkage策略, 将协方差矩阵向单位矩阵缩并
    Sigma = (1 - beta) * Sigma + beta * np.eye(n_features)
    # 如果协方差矩阵奇异(只有alpha是默认值0时才有可能), 就强制缩并
    if np.linalg.matrix_rank(Sigma) < n_features:
        Sigma = (1-1.0e-4)*Sigma + 1.0e-4*np.eye(n_features)
    Sigma_inv = np.linalg.inv(Sigma)

    ## II. 测试, 评估对测试样本的标签做预测
    # 定义线性判别函数
    def gc_Xi(Xi, Mu_c, Sigma_inv, prior_c):
        Mu_c.shape = (Mu_c.shape[0], 1)  # np的一维数组转成二维才能做转置
        w = np.dot(Sigma_inv, Mu_c)
        b = -0.5*np.dot(np.dot(np.transpose(Mu_c),Sigma_inv), Mu_c) + np.log(prior_c)
        return np.dot(np.transpose(w), Xi) + b
    # 预测测试样本的标签, 计算正确率
    accuracy = 0
    n_testsamples = test_X.shape[0]
    g_Xi = np.zeros(shape=(C,))
    # test_X = test_X / Var
    for i in range(n_testsamples):
        Xi = test_X[i]
        for c in range(C):
            g_Xi[c] = gc_Xi(Xi, Mu[c], Sigma_inv, prior[c])
        if np.where(g_Xi==max(g_Xi))[0] == test_y[i]:
            accuracy += 1
    accuracy /= n_testsamples
    return accuracy

def quadraticDF(train_X, train_y, test_X, test_y, alpha=0.0, prior=None):
    """
    二次判别函数QDF, 包括训练和测试
    :param train_X: 全部训练样本的特征矢量
    :param train_y: 全部训练样本的标签
    :param test_X: 全部测试样本的特征矢量
    :param test_y: 全部测试样本的标签
    :param alpha: Shrinkage缩并策略的参数值
    :param prior: 各个类的先验概率, 如不指定将自行计算
    :return: 模型在测试集上的accuracy
    """
    n, n_features = train_X.shape
    ## 计算各个类别的先验概率
    classLabel, y = np.unique(train_y, return_inverse=True)
    n_i = np.bincount(y)  # 各个类别的样本数
    if prior is None:
        prior = np.bincount(y) / float(n)
    C = len(classLabel)

    # 预先求出Shrinkage策略中要用到的共同协方差矩阵
    Var = np.var(train_X, axis=0)
    for i in range(n_features):
        if Var[i] == 0:
            Var[i] = 1.0e-4
    Sigma_All = np.cov(train_X / Var, rowvar=False)
    if np.linalg.matrix_rank(Sigma_All) < n_features:
        Sigma_All = (1 - 1.0e-4)*Sigma_All + 1.0e-4*np.eye(n_features)

    ## I. 训练, 使用MLE来估计各个类别的均值向量和协方差矩阵, 并求协方差矩阵的逆矩阵
    Mu = np.zeros(shape=(C, n_features))
    Sigma = np.zeros(shape=(C, n_features, n_features))
    Sigma_inv = np.zeros(shape=(C, n_features, n_features))
    for c in classLabel:
        train_Xc = train_X[y==c, :]
        Mu[c] = np.mean(train_Xc, axis=0)
        Sigma[c] = np.cov(train_Xc - Mu[c], rowvar=False)
        # Shrinkage策略, 将协方差矩阵向同一矩阵缩并
        Sigma[c] = ((1 - alpha)*n_i[c]*Sigma[c] + alpha*n*Sigma_All) / ((1 - alpha)*n_i[c] + alpha*n)
        # 如果协方差矩阵奇异, 就强制缩并
        if np.linalg.matrix_rank(Sigma[c]) < n_features:
            alpha = 1.0e-4
            Sigma[c] = ((1 - alpha)*n_i[c]*Sigma[c] + alpha*n*Sigma_All) / ((1 - alpha)*n_i[c] + alpha*n)
        Sigma_inv[c] = np.linalg.inv(Sigma[c])

    ## II. 测试, 评估对测试样本的标签做预测
    # 定义二次判别函数
    def gc_Xi(Xi, Mu_c, Sigma_c, Sigma_inv_c, prior_c):
        Mu_c.shape = (Mu_c.shape[0], 1)  # np的一维数组转成二维才能做转置
        W = -0.5 * Sigma_inv_c
        w = np.dot(Sigma_inv_c, Mu_c)
        # 矩阵太大, 直接用np.linalg.det()会overflow, 用sign*np.exp(logdet)也会overflow. 这里直接求了行列式的ln, 避开了overflow
        (sign, logdet) = np.linalg.slogdet(Sigma_c)
        ln_det_Sigma_c = np.log(sign) + logdet
        b = -0.5*np.dot(np.dot(np.transpose(Mu_c),Sigma_inv_c), Mu_c) - 0.5*ln_det_Sigma_c + np.log(prior_c)
        return np.dot(np.dot(np.transpose(Xi), W), Xi) + np.dot(np.transpose(w), Xi) + b
    # 预测测试样本的标签, 计算正确率
    accuracy = 0
    n_testsamples = test_X.shape[0]
    g_Xi = np.zeros(shape=(C,))
    for i in range(n_testsamples):
        Xi = test_X[i]
        for c in range(C):
            g_Xi[c] = gc_Xi(Xi, Mu[c], Sigma[c], Sigma_inv[c], prior[c])
        if np.where(g_Xi==max(g_Xi))[0] == test_y[i]:
            accuracy += 1
    accuracy /= n_testsamples
    return accuracy

def validationPara(model, train_X, train_y, para_init, para_max, step):
    """
    超参数调节, 将训练集的十分之一作为验证集
    :param model: 模型, LDF或QDF
    :param train_X: 训练集的特征矢量
    :param train_y: 训练集的标签
    :param para_init: 超参数初值
    :param para_max: 超参数最大值
    :param step: 超参数更新步长
    :return: para_choose: 选定的超参数
             para: 尝试过的超参数取值
             target: 各para在验证集上的效果
    """
    para_choose = para_init; para = []
    m = int(train_X.shape[0] * 0.9)  # 训练数据的样本数
    max_target = 0; target = []
    while para_init <= para_max:
        para.append(para_init)
        target_v = model(train_X[ :m], train_y[ :m], train_X[m: ], train_y[m: ], para_init)
        target.append(target_v)
        if target_v > max_target:
            para_choose = para_init
            max_target = target_v
        para_init = round(para_init + step, 2)
    return para_choose, para, target

def experimentsVisualize(para, target, title, filename):
    """
    可视化调参过程中, 验证集上的效果
    :param para: 尝试过的超参数取值
    :param target: 各para在验证集上的效果
    :return: 无
    """
    target = 100 * np.array(target)
    fig1 = plt.figure()  # figsize=(10, 8)
    ax1 = fig1.add_subplot(111)
    ax1.plot(para, target, marker = 's', c='red', lw=1.5, label='$Accuracy$')
    index = np.where(target == max(target))[0][0]
    plt.text(para[index]-0.03, target[index]+1, str(round(target[index],2)))
    plt.legend(loc=4)
    plt.xlabel('$' + title + '$')
    plt.ylabel('$Accuracy(\%)$')
    plt.title('$Accuracy(\%)$' + ' on validation-set with ' + 'Hyperparameter $' + title + '$')
    plt.ylim(60, 100)
    plt.xlim(min(para)-0.05, max(para)+0.05)
    plt.savefig(filename + '.pdf', dpi=400)
    # plt.show()

if __name__=='__main__':
    train_X, train_y, test_X, test_y = inputData()
    train_X, test_X = pcaFeatures(train_X, test_X, 50)
    ## (一) 线性判别函数LDF
    # 1. 在验证集上调参
    beta_choose, beta, accuracy = validationPara(linearDF, train_X, train_y, 0.0, 1.0, 0.05)
    # 2. 选取最优超参数后, 训练模型并测试
    accuracy_final = linearDF(train_X, train_y, test_X, test_y, beta=beta_choose)
    print('LDF: beta = %.2f, accuracy = %.2f%%' % (beta_choose, 100 * accuracy_final))
    experimentsVisualize(beta, accuracy, r'\beta', '5_9')
    # 3. 与sklearn库的结果进行对比
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDF
    from sklearn.metrics import  accuracy_score
    ldf = LDF(solver='lsqr')  # sklearn的LDF中的shrinkage有一点区别
    ldf.fit(train_X, train_y)
    accuracy_final = 100 * accuracy_score(test_y, ldf.predict(test_X))
    print('LDF(sklearn):     accuracy = %.2f%%' % accuracy_final)

    ## (二) 二次判别函数QDF
    # 1. 在验证集上调参
    alpha_choose, alpha, accuracy = validationPara(quadraticDF, train_X, train_y, 0.0, 1.0, 0.05)
    # 2. 选取最优超参数后, 训练模型并测试
    accuracy_final = quadraticDF(train_X, train_y, test_X, test_y, alpha=alpha_choose)
    print('QDF: alpha = %.2f, accuracy = %.2f%%' % (alpha_choose, 100 * accuracy_final))
    experimentsVisualize(alpha, accuracy, r'\alpha', '5_10')
    # 3. 与sklearn库的结果进行对比, sklearn的速度明显要快非常多
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDF
    qdf = QDF()  # sklearn的QDF中的shrinkage是直接向单位矩阵所并
    qdf.fit(train_X, train_y)
    accuracy_final = 100 * accuracy_score(test_y, qdf.predict(test_X))
    print('QDF(sklearn):      accuracy = %.2f%%' % accuracy_final)
