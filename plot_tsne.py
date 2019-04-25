    # _*_ coding:utf-8 -*-
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

# kitting = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/sensor_info_no_recovery_skill_pos.npy')
# tag = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/tag_info_no_recovery_skill_pos.npy')

<<<<<<< HEAD
# 加载数据集
iris = load_iris()
kitting = np.load('/home/jim/Recovery_reimplement/sensor_info_no_recovery_skill_pos.npy')
tag = np.load('/home/jim/Recovery_reimplement/tag_info_no_recovery_skill_pos.npy')
=======
kitting = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/sensor_info_no_recovery_skill_pos_et.npy')
tag = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/tag_info_no_recovery_skill_pos_et.npy')
>>>>>>> 88c0f102909b2bfe61d94eb2b8a9f285a9adf362

colors = {3:'r',4:'g',5:'b',7:'c',8:'m',9:'y'
}

colors = {3:'black',
            4:'g',
            5:'b',
            7:'c',
            8:'m',
            9:'y'
                }

print(kitting.shape)

<<<<<<< HEAD
# 对应的标签有0,1,2三种
# print(iris.target.shape)
# 使用TSNE进行降维处理
# tsne = TSNE(n_components=2, learning_rate=100).fit_transform(iris.data)
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(kitting)
# 使用PCA 进行降维处理
# pca = PCA().fit_transform(iris.data)
# pca = PCA().fit_transform(kitting)
# # 设置画布的大小
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.scatter(tsne[:, 0], tsne[:, 1], c=iris.target)
# plt.scatter(tsne[:, 0], tsne[:, 1], c=colors[tag], label='Skill'+str(tag))
for i in range(len(tsne)):
    plt.scatter(tsne[:, 0], tsne[:, 1], c=tag)
    plt.plot(c=tag, label='Skill'+str(tag))
# plt.subplot(122)
# plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
# plt.scatter(pca[:, 0], pca[:, 1])

plt.colorbar()
plt.show()
=======
tsne = TSNE(n_components=3, learning_rate=100).fit_transform(kitting)
a = tsne
np.save("tsne.npy", a)


# fig = plt.figure()


# ax = fig.gca(projection='3d', adjustable='box')

# # plt.scatter(tsne[:, 0], tsne[:, 1], c=colors[t], label ='Skill ' + str(1), alpha=0.5)
# # plt.scatter(tsne[:, 0], tsne[:, 1], c=tag, label ='Skill ' + str(1), alpha=0.5)
# ax = ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=tag*20, label ='Skill ' + str(tag), alpha=0.5)


# plt.legend()

# # for tg, p in zip(tag, tsne):
# #     ax.scatter(p[0], p[1], p[2], c=colors[tg], label ='Skill ' + str(tg), alpha=0.55)

# # handles, labels = plt.gca().get_legend_handles_labels()
# # by_label = OrderedDict(zip(labels, handles))
# # plt.legend(by_label.values(), by_label.keys(), loc=1)
# # fig.savefig('tsne.png', format="png", dpi=300)
# # plt.colorbar()
# plt.show()
>>>>>>> 88c0f102909b2bfe61d94eb2b8a9f285a9adf362
