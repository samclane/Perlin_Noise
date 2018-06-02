from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def brownian(delta, dt, x, n):
    for k in range(n):
        x = x + norm.rvs(scale=delta**2*dt)
        yield x


def fast_brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def main():
    x = [v for v in brownian(0.25, .1, 0.0, 50)]
    plt.plot(x)
    plt.show()

if __name__ == "__main__":
    main()