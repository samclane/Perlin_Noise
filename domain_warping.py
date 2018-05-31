import numpy as np
import math
from array2gif import write_gif

""" Based on shaders from https://thebookofshaders.com/13/ """

u_resolution = np.zeros(2)
u_mouse = np.zeros(2)
u_time = 0.0

def random(_st):
    return math.modf(math.sin(np.dot(_st, np.array([12.9898,78.233])*43758.5453123)))[0]

def mix(x, y, a):
    return x * (1 - a) + y * a

def clamp(x, minVal, maxVal):
    return min(max(x, minVal), maxVal)

def noise(_st):
    copy_st = _st.copy()
    i = np.vectorize(lambda x: math.floor(x))(copy_st)
    f = np.vectorize(lambda x: math.modf(x)[0])(copy_st)

    a = random(i)
    b = random(i + np.array([1.0, 0.0]))
    c = random(i + np.array([0.0, 1.0]))
    d = random(i + np.array([1.0, 1.0]))

    u = f * f * (3.0 - 2.0 * f)

    return mix(a, b, u[0]) + (c - a) * u[1] * (1.0 - u[0]) + (d - b) * u[0] * u[1]

NUM_OCTAVES = 5

def fbm(_st):
    v = 0.0
    a = 0.5
    shift = np.array([100.0, 0.0])
    # Rotate to reduce axial bias
    rot = np.array([[math.cos(0.5), math.sin(0.5)], [-1*math.sin(0.5), math.cos(0.5)]])
    for i in range(0, NUM_OCTAVES):
        v += a * noise(_st)
        _st = (rot @ _st) * 2.0 + shift
        a *= 0.5
    return v

gl_FragCoord = u_resolution.copy()
viewport = np.zeros((20, 20, 3))

def main():
    for x in range(0,20):
        for y in range(0, 20):
            st = gl_FragCoord
            color = np.array([0.0, 0.0, 0.0])

            q = np.array([0.0, 0.0])
            q[0] = fbm(st + 0.00*u_time)
            q[1] = fbm(st + np.array([1.0, 0.0]))

            r = np.array([0.0, 0.0])
            r[0] = fbm(st + 1.0*q + np.array([1.7, 9.2]) + 0.15*u_time)
            r[1] = fbm(st + 1.0*q + np.array([8.3, 2.8]) + 0.126*u_time)

            f = fbm(st+r)

            color = mix(np.array([0.101961,0.619608,0.666667]),
                        np.array([0.666667,0.666667,0.498039]),
                        clamp((f * f) * 4.0, 0.0, 1.0))

            color = mix(color,
                        np.array([0, 0, 0.164706]),
                        clamp(math.hypot(*q), 0.0, 1.0))

            color = mix(color,
                        np.array([0.666667,1,1]),
                        clamp(math.sqrt(r[0]**2), 0.0, 1.0))

            viewport[x,y] = np.vectorize(lambda x: int(255*abs(x)))(np.array([*((f*f*f+.6*f*f+.5*f)*color)]))
    write_gif(viewport, 'test.gif', fps=5)


if __name__ == "__main__":
    main()