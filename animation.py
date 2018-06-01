import noise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

RES = 50
SIZE = 200

frames = 400
pixels = None

def animate(i):
    im = plt.imshow(pixels[:, :, i])

    return im

if __name__ == "__main__":
    pixels = np.zeros((SIZE, SIZE, frames))
    for f in range(frames):
        for xp in range(SIZE):
            for yp in range(SIZE):
                pixels[xp, yp, f] = noise.pnoise3(xp / RES, yp / RES, f / RES)

    fig = plt.figure()
    ax = plt.axes()
    anim = animation.FuncAnimation(fig, animate, frames=frames)
    print('rendering...')
    anim.save('animation.gif', writer='imagemagick', fps=60)
    print('done')