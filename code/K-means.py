import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
import time

def cal_dis(x, y):  # 计算两点之间的距离
    return np.sqrt(np.sum(np.power((x - y), 2)))

class kmeans:
    def __init__(self, num_clusters, train_data_path, test_data_path):
        self.num_clusters = num_clusters  # 聚类中心个数
        train_data = pd.read_csv(train_data_path).values  # 读取数据
        test_data = pd.read_csv(test_data_path).values
        self.train_labels = train_data[:, 0]
        self.train_features = train_data[:, 1:]
        self.test_labels = test_data[:, 0]
        self.test_features = test_data[:, 1:]
        pca = PCA(n_components=200)
        pca.fit(self.train_features)
        self.train_features = pca.transform(self.train_features)
        self.test_features = pca.transform(self.test_features)
        self.cent = self.init_cent('++')  # 初始化聚类中心
        self.cent_num = {}  # 记录每个簇对应的数字，用于测试正确率

    def init_cent(self, mode):
        if mode == 'random':  # 随机初始化聚类中心
            random_ind = random.sample(range(len(self.train_features)), self.num_clusters)
            cent = self.train_features[random_ind]
            return cent
        elif mode == '++':  # K-Means++
            random_ind = random.sample(range(len(self.train_features)), 1)  # 随机选一个初始点
            cent = np.array(self.train_features[random_ind]).reshape(1, -1)
            while np.shape(cent)[0] < self.num_clusters:
                dis = np.full(len(self.train_labels), np.inf)
                for c in cent:  # 计算所有点到与之最近的聚类中心的距离，选取最大距离的点加入到新的聚类中心集合中
                    dis = np.minimum(np.sum(np.power((self.train_features - c), 2), axis=1), dis)
                cent = np.append(cent, self.train_features[np.argmax(dis)].reshape(1, -1), axis=0)
            return cent

    def cal_clusters(self, x):  # 输入一个点x，计算出离这个点最近的聚类中心的下标
        return np.argmin(np.sum(np.power((self.cent - x), 2), axis=1))

    def cal_new_cent(self, clusters):  # 根据当前分类情况，计算出新的聚类中心
        for i, clu in enumerate(clusters):
            self.cent[i] = np.mean(np.array(clu), axis=0)

    def train(self):
        epochs = 0
        while True:
            epochs += 1
            # print(epochs)
            old_cent = np.copy(self.cent)  # 记录下更新前的聚类中心
            clusters = [[] for _ in range(self.num_clusters)]  # 第n行包含的点代表被分到第n个簇中的点
            clusters_num = [[] for _ in range(self.num_clusters)]
            for i, x in enumerate(self.train_features):  # 遍历每个样本点，计算出其所属的簇
                owner_ind = self.cal_clusters(x)
                clusters[owner_ind].append(x)
                clusters_num[owner_ind].append(self.train_labels[i])
            self.cal_new_cent(clusters)  # 更新聚类中心
            if np.allclose(old_cent, self.cent):  # 如果更新前后的聚类中心差别不大，代表算法已收敛
                for i, x in enumerate(clusters_num):
                    self.cent_num[i] = np.argmax(np.bincount(x))  # 计算出每个簇中包含最多的数字，将其作为簇对应的数字，用于测试正确率
                print("训练完成, epochs=%d" % epochs)
                break

    def test(self):
        acc = 0
        for i, x in enumerate(self.test_features):  # 遍历每个测试样本，计算准确率
            owner = self.cent_num[self.cal_clusters(x)]
            if owner == self.test_labels[i]:
                acc += 1
        acc = acc / len(self.test_labels) * 100
        print("准确率: %.2f %%" % acc)
        return acc

if __name__ == '__main__':
    k = kmeans(10, '../data/mnist_train.csv', '../data/mnist_test.csv')
    acc = 0
    n = 10
    for _ in range(n):
        t1 = time.time()
        k.cent = k.init_cent('++')
        k.train()
        t2 = time.time()
        acc = max(acc, k.test())
        print("耗时:" + str(t2 - t1) + "s = " + str((t2 - t1)/60) + "mins")
    print('%d次的最佳正确率: %.2f %%' % (n, acc))

