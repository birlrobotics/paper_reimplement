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

kitting = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/sensor_info_no_recovery_skill_pos_et.npy')
tag = np.load('/home/birl-spai-ubuntu14/baxter_ws/src/SPAI/Recovery_reimplement/tag_info_no_recovery_skill_pos_et.npy')


colors = {3:'black',
            4:'g',
            5:'b',
            7:'c',
            8:'m',
            9:'y'
                }

print(kitting.shape)

tsne = TSNE(n_components=3, learning_rate=100).fit_transform(kitting)

# plt.scatter(tsne[:, 0], tsne[:, 1], c=colors[t], label ='Skill ' + str(1), alpha=0.5)
# plt.scatter(tsne[:, 0], tsne[:, 1], c=tag, label ='Skill ' + str(1), alpha=0.5)
fig = plt.figure()
ax = fig.gca(projection='3d', adjustable='box')

for tg, p in zip(tag, tsne):
    ax.scatter(p[0], p[1], p[2], c=colors[tg], label ='Skill ' + str(tg), alpha=0.55)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc=1)
fig.savefig('tsne.png', format="png", dpi=300)
# plt.colorbar()
plt.show()
