# _*_ coding:utf-8 -*-
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class data():
    def __init__(self, data, target):
        self.data = data
        self.target = target

kitting = np.load('/home/jim/Recovery_reimplement/sensor_info_no_recovery_skill_pos.npy')
tag = np.load('/home/jim/Recovery_reimplement/tag_info_no_recovery_skill_pos.npy')


print(kitting.shape)

for j in range(20):
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(kitting)

    for i in range(len(tsne)):
        plt.scatter(tsne[:, 0], tsne[:, 1], c=colors[tag])
        plt.plot(c=colors[tag], label='Skill'+str(tag))

    # plt.subplot(122)
    # plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
    # plt.scatter(pca[:, 0], pca[:, 1])

    # plt.colorbar()
    # plt.show()
    plt.savefig('all_points_tsne_'+str(j)+'.pdf', dpi=300)
