import noise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from brownian import fast_brownian
import time

RES = 25
SIZE = 200
BROWNIAN = True

frames = 60
pixels = None

def animate(i):
    im = plt.imshow(pixels[:, :, i])

    return im

if __name__ == "__main__":
    pixels = np.zeros((SIZE, SIZE, frames))
    brownian_z_slice = fast_brownian(0, frames, 0.1, 0.25)
    print('generating slices...')
    for f in range(frames):
        for xp in range(SIZE):
            for yp in range(SIZE):
                pixels[xp, yp, f] = noise.pnoise3(xp / RES, yp / RES, brownian_z_slice[f] if BROWNIAN else f / RES)
    print('done')
    fig = plt.figure()
    ax = plt.axes()
    anim = animation.FuncAnimation(fig, animate, frames=frames)
    print('rendering...')
    anim.save('animation-{}.gif'.format(time.time()), writer='imagemagick', fps=60)
    print('done')