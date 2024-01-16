import numpy as np
from scipy import stats
import pandas as pd
import random
from sklearn.decomposition import PCA
import time

class GMM(object):
    def __init__(self, num_clusters, train_data_path, test_data_path, mu_init_mode, sigma_init_mode):
        self.p = None
        self.mu = None
        self.sigma = None
        self.num_clusters = num_clusters  # 高斯分布个数
        self.own = np.empty(self.num_clusters)  # 每个高斯分布中概率最大的一个点对应的标签
        train_data = pd.read_csv(train_data_path).values  # 读取数据
        test_data = pd.read_csv(test_data_path).values
        self.train_labels = train_data[:, 0]
        self.train_features = train_data[:, 1:]/255.0
        self.test_labels = test_data[:, 0]
        self.test_features = test_data[:, 1:]/255.0
        pca = PCA(n_components=50)  # 使用PCA进行降维
        pca.fit(self.train_features)
        self.train_features = pca.transform(self.train_features)
        self.test_features = pca.transform(self.test_features)
        self.init(mu_init_mode, sigma_init_mode)  # 初始化参数

    def init(self, mu_init_mode, sigma_init_mode):
        self.p = np.random.uniform(low=1e-7, high=1, size=self.num_clusters)
        self.p = self.p / self.p.sum()  # 保证所有p_k的和为1
        if mu_init_mode == 'random':  # 随机取k个样本点作为高斯的均值
            random_ind = random.sample(range(len(self.train_features)), self.num_clusters)
            self.mu = self.train_features[random_ind]
        elif mu_init_mode == '++':  # K-Means++初始化法
            random_ind = random.sample(range(len(self.train_features)), 1)  # 随机选一个初始点
            cent = np.array(self.train_features[random_ind]).reshape(1, -1)
            while np.shape(cent)[0] < self.num_clusters:
                dis = np.full(len(self.train_labels), np.inf)
                for c in cent:  # 计算所有点到与之最近的已选点的距离，选取最大距离的点加入到均值集合中
                    dis = np.minimum(np.sum(np.power((self.train_features - c), 2), axis=1), dis)
                cent = np.append(cent, self.train_features[np.argmax(dis)].reshape(1, -1), axis=0)
            self.mu = cent
        self.sigma = np.empty((self.num_clusters, len(self.train_features[0]), len(self.train_features[0])))
        for i in range(self.num_clusters):  # 随机生成协方差矩阵
            if sigma_init_mode == 'diag_equal':  # 对角矩阵，且值都相同
                self.sigma[i] = np.eye(len(self.train_features[0])) * (random.random() + 1e-8)
            elif sigma_init_mode == 'diag_not_equal':  # 对角矩阵，但值不同
                self.sigma[i] = np.diag(np.random.rand(len(self.train_features[0])) + 1e-8) * 0.8
            elif sigma_init_mode == 'random':  # 普通的对称正定矩阵
                A = np.random.rand(len(self.train_features[0]), len(self.train_features[0]))
                self.sigma[i] = np.dot(A.T, A) * 0.1 + np.eye(len(self.train_features[0])) * 0.1
            self.sigma[i] += np.eye(len(self.train_features[0])) * 1e-7
    def train(self):
        epoch = 0
        while True:
            epoch += 1
            # E步
            old_mu = self.mu
            old_p = self.p
            old_sigma = self.sigma
            phi = np.zeros((self.train_features.shape[0], self.num_clusters))
            for i in range(self.num_clusters):
                # 生成K个概率密度函数并计算对于所有样本的概率密度
                a = self.p[i] * stats.multivariate_normal.pdf(self.train_features, mean=self.mu[i], cov=self.sigma[i])
                phi[:, i] = a
            # 计算所有样本属于每一类别的后验
            gamma = phi / phi.sum(axis=1).reshape(-1, 1)
            # M步
            # 计算下一时刻的参数值
            p_hat = gamma.sum(axis=0)  # 目前是phi_ij对i求和
            mu_hat = np.tensordot(gamma, self.train_features, axes=[0, 0]) / p_hat.reshape(-1, 1)  # 表示让gamma的第0维(列)和x的第0维(列)作内积
            sigma_hat = np.empty(self.sigma.shape)
            for i in range(self.num_clusters):
                tmp = self.train_features - self.mu[i]
                sigma_hat[i] = np.dot(tmp.T * gamma[:, i], tmp)
                sigma_hat[i] /= p_hat[i]
            # 更新参数
            self.sigma = sigma_hat + np.eye(len(self.train_features[0])) * 1e-7
            self.mu = mu_hat
            p_hat = p_hat / len(self.train_features)
            self.p = p_hat
            # 如果已收敛，则退出
            if np.allclose(old_p, self.p, atol=1e-4) and np.allclose(old_mu, self.mu, atol=1e-4) and np.allclose(old_sigma, self.sigma, atol=1e-4):
                print('epoch:%d' % epoch)
                max_ind = np.argmax(gamma, axis=0)
                self.own = self.train_labels[max_ind]
                break

    def test(self):
        acc = 0
        for i, x in enumerate(self.test_features):  # 遍历每个测试样本，计算准确率
            p = [-1 for _ in range(self.num_clusters)]
            for j in range(self.num_clusters):
                p[j] = stats.multivariate_normal.pdf(x, self.mu[j], self.sigma[j])
            owner = self.own[np.argmax(p)]
            if owner == self.test_labels[i]:
                acc += 1
        acc = acc / len(self.test_labels) * 100
        print("准确率: %.2f %%" % acc)
        return acc

if __name__ == '__main__':
    mu_init_mode = 'random'
    # mu_init_mode = '++'
    sigma_init_mode = 'diag_equal'
    # sigma_init_mode = 'diag_not_equal'
    # sigma_init_mode = 'random'
    gmm = GMM(10, '../data/mnist_train.csv', '../data/mnist_test.csv', mu_init_mode, sigma_init_mode)
    for i in range(10):
        if i != 0:
            gmm.init(mu_init_mode, sigma_init_mode)
        t1 = time.time()
        gmm.train()
        t2 = time.time()
        gmm.test()
        print("耗时: %.3f s = %.3f mins" % ((t2 - t1), (t2 - t1)/60))