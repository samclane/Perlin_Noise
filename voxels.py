import noise
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d projection

RES = 5
SIZE = 20
voxels = np.zeros((SIZE,SIZE,SIZE), dtype=np.bool)

for xp in range(SIZE):
    for yp in range(SIZE):
        for zp in range(SIZE):
            voxels[xp,yp,zp] = True if abs(noise.pnoise3(xp/RES,yp/RES,zp/RES)) > .5 else False

colors = np.empty(voxels.shape, dtype=object)
colors[voxels] = 'green'

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')
plt.show()