import numpy as np
import clustering
import matplotlib.pyplot as plt


s1 = np.random.uniform(0, 5, (100,3))
s2 = np.random.uniform(2, 7, (100,3))
s3 = np.random.uniform(100, 300, (100,3))
state = np.concatenate((s1, s2, s3), axis=0)
ax = plt.subplot(111, projection='3d')
ax.scatter(s1[:0], s1[:1], s1[:2], c='y')
ax.scatter(s2[:0], s2[:1], s2[:2], c='r')
ax.scatter(s3[:0], s3[:1], s3[:2], c='g')

ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()