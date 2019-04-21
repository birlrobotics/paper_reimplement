# _*_ coding:utf-8 -*-
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

class data():
    def __init__(self, data, target):
        self.data = data
        self.target = target

# 加载数据集
iris = load_iris()
kitting = np.load('/home/jim/Recovery_reimplement/sensor_info_no_recovery_skill_pos.npy')

# # 共有150个例子， 数据的类型是numpy.ndarray
# print(iris.data.shape)
print(kitting.shape)

# 对应的标签有0,1,2三种
# print(iris.target.shape)
# 使用TSNE进行降维处理
# tsne = TSNE(n_components=2, learning_rate=100).fit_transform(iris.data)
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(kitting)
# 使用PCA 进行降维处理
# pca = PCA().fit_transform(iris.data)
# pca = PCA().fit_transform(kitting)
# # 设置画布的大小
plt.figure(figsize=(12, 6))
plt.subplot(121)
# plt.scatter(tsne[:, 0], tsne[:, 1], c=iris.target)
plt.scatter(tsne[:, 0], tsne[:, 1])

plt.subplot(122)
# plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
plt.scatter(pca[:, 0], pca[:, 1])

# plt.colorbar()
plt.show()