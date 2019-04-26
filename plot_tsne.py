    # _*_ coding:utf-8 -*-
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import time

kitting = np.load('/home/jim/Recovery_reimplement/sensor_info_no_recovery_skill_pos.npy')
tag = np.load('/home/jim/Recovery_reimplement/tag_info_no_recovery_skill_pos.npy')

# kitting = np.load('/home/jim/Recovery_reimplement/sensor_info_no_recovery_skill_pos_et.npy')
# tag = np.load('/home/jim/Recovery_reimplement/tag_info_no_recovery_skill_pos_et.npy')


colors = {3:'black',
            4:'g',
            5:'b',
            7:'c',
            8:'m',
            9:'y'
                }

print(kitting.shape)

start = time.clock()

# for i in range(20):
#     tsne = TSNE(n_components=2, learning_rate=100).fit_transform(kitting)
#     a = tsne

#     np.save("tsne"+str(i), a)

#     # # plt.scatter(tsne[:, 0], tsne[:, 1], c=colors[t], label ='Skill ' + str(1), alpha=0.5)
#     plt.scatter(tsne[:, 0], tsne[:, 1], c=tag, label ='Skill ' + str(1), alpha=0.5)

#     # fig = plt.figure()
#     # ax = fig.gca(projection='3d', adjustable='box')
#     # ax = ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=tag*20, label ='Skill ' + str(tag), alpha=0.5)

#     # # for tg, p in zip(tag, tsne):
#     # #     ax.scatter(p[0], p[1], p[2], c=colors[tg], label ='Skill ' + str(tg), alpha=0.55)

#     # # handles, labels = plt.gca().get_legend_handles_labels()
#     # # by_label = OrderedDict(zip(labels, handles))
#     # # plt.legend(by_label.values(), by_label.keys(), loc=1)
#     # # fig.savefig('tsne.png', format="png", dpi=300)
#     # # plt.colorbar()
#     plt.legend()
#     # plt.show()
#     plt.savefig('tsne'+str(i), format="png", dpi=300)


tsne = np.load('/home/jim/Recovery_reimplement/tsne0.npy')
# tsne = np.load('/home/jim/Recovery_reimplement/tsne_499.npy')

# # plt.scatter(tsne[:, 0], tsne[:, 1], c=colors[t], label ='Skill ' + str(1), alpha=0.5)
# plt.scatter(tsne[:, 0], tsne[:, 1], c=tag, label ='Skill ' + str(1), alpha=0.5)

# fig = plt.figure()
# ax = fig.gca(projection='3d', adjustable='box')
# ax = ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=tag*20, label ='Skill ' + str(tag), alpha=0.5)

for tg, p in zip(tag, tsne):
    plt.scatter(p[0], p[1], c=colors[tg], label ='Skill ' + str(tg), alpha=0.55)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc=1)
# # fig.savefig('tsne.png', format="png", dpi=300)
# # plt.colorbar()
plt.legend()
plt.show()
# plt.savefig('tsne'+str(i), format="png", dpi=300)



end = time.clock()
print end-start 
