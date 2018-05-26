import math
from collections import namedtuple
from functools import lru_cache, wraps
from random import randint
from time import time

import numpy as np
import scipy.misc as smp

# Settings
enable_lru = True
enable_color = True
repeat = 0  # not sure really what this does. it was in the tutorial. it breaks color.

p = [151, 160, 137, 91, 90, 15,
     131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
     190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
     88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
     77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
     102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
     135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
     5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
     223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
     251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
     49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
     138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180]
p.extend(p)

Vector = namedtuple('Vector', 'x y z')


def use_lru(func):
    if not enable_lru:
        return func

    @lru_cache(maxsize=None)
    def decorator(*args):
        return func(*args)

    return decorator


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return float('.'.join([i, (d + '0' * n)[:n]]))


def truncate_args(digits):
    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            args = (truncate(x, digits) for x in args)
            return func(*args)

        return wrapper

    return decorator


@use_lru
def fade(t):
    return (t ** 3) * (t * (t * 6 - 15) + 10)


@use_lru
def inc(num):
    num += 1
    if repeat > 0:
        num %= repeat
    return num


@use_lru
def hash_row(x, y, z):
    return p[p[p[x] + y] + z]


def grad_slow(hash, x, y, z):
    h = hash & 15
    u = x if h < 8 else y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def grad(hash, x, y, z):
    switch = hash & 0xF
    if switch == 0x0:
        return x + y
    elif switch == 0x1:
        return -x + y
    elif switch == 0x2:
        return x - y
    elif switch == 0x3:
        return -x - y
    elif switch == 0x4:
        return x + z
    elif switch == 0x5:
        return -x + z
    elif switch == 0x6:
        return x - z
    elif switch == 0x7:
        return -x - z
    elif switch == 0x8:
        return y + z
    elif switch == 0x9:
        return -y + z
    elif switch == 0xA:
        return y - z
    elif switch == 0xB:
        return -y - z
    elif switch == 0xC:
        return y + x
    elif switch == 0xD:
        return -y + z
    elif switch == 0xE:
        return y - x
    elif switch == 0xF:
        return -y - z


def lerp(a, b, x):
    return a + x * (b - a)


def perlin(x, y, z):
    arglist = Vector(x, y, z)
    if repeat > 0:
        arglist = Vector(*map(lambda x: x % repeat, arglist))
    float_args, int_args = [Vector(*l) for l in zip(*list(map(math.modf, arglist)))]
    int_args = Vector(*map(int, int_args))

    fade_vector = Vector(*map(fade, float_args))

    aaa = hash_row(*int_args)
    aba = hash_row(int_args.x, inc(int_args.y), int_args.z)
    aab = hash_row(int_args.x, int_args.y, inc(int_args.z))
    abb = hash_row(int_args.x, inc(int_args.y), inc(int_args.z))
    baa = hash_row(inc(int_args.x), int_args.y, int_args.z)
    bba = hash_row(inc(int_args.x), inc(int_args.y), int_args.z)
    bab = hash_row(inc(int_args.x), int_args.y, inc(int_args.z))
    bbb = hash_row(*map(inc, int_args))

    x1 = lerp(grad(aaa, *float_args), grad(baa, float_args.x - 1, float_args.y, float_args.z), fade_vector.x)
    x2 = lerp(grad(aba, float_args.x, float_args.y - 1, float_args.z),
              grad(bba, float_args.x - 1, float_args.y - 1, float_args.z), fade_vector.x)
    y1 = lerp(x1, x2, fade_vector.y)
    x1 = lerp(grad(aab, float_args.x, float_args.y, float_args.z - 1),
              grad(bab, float_args.x - 1, float_args.y, float_args.z - 1), fade_vector.x)
    x2 = lerp(grad(abb, float_args.x, float_args.y - 1, float_args.z - 1),
              grad(bbb, float_args.x - 1, float_args.y - 1, float_args.z - 1), fade_vector.x)
    y2 = lerp(x1, x2, fade_vector.y)

    return (lerp(y1, y2, fade_vector.z) + 1) / 2


def octave_perlin(x, y, z, octaves, persistence):
    total = 0
    frequency = 1
    amplitude = 1
    maxValue = 0
    for i in range(octaves):
        total += perlin(x * frequency, y * frequency, z * frequency) * amplitude
        maxValue += amplitude
        amplitude += persistence
        frequency *= 2
    return total / maxValue


# Size of the screen
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300

# how fine the noise is. lower => finer features
UNIT_CUBE = 128


def main():
    starttime = time()
    data = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)

    # choose a random z-slice to get a random image back. otherwise perlin() always returns the same map (z=0)
    z = randint(1, UNIT_CUBE)
    colorseed = randint(1, UNIT_CUBE), randint(1, UNIT_CUBE), randint(1, UNIT_CUBE)

    for x in range(SCREEN_WIDTH):
        for y in range(SCREEN_HEIGHT):
            value = perlin(x // UNIT_CUBE + (x % UNIT_CUBE) / UNIT_CUBE, y // UNIT_CUBE + (y % UNIT_CUBE) / UNIT_CUBE,
                           z)
            if enable_color:
                r = perlin(x // UNIT_CUBE + (x % UNIT_CUBE) / UNIT_CUBE, y // UNIT_CUBE + (y % UNIT_CUBE) / UNIT_CUBE,
                           colorseed[0])
                g = perlin(x // UNIT_CUBE + (x % UNIT_CUBE) / UNIT_CUBE, y // UNIT_CUBE + (y % UNIT_CUBE) / UNIT_CUBE,
                           colorseed[1])
                b = perlin(x // UNIT_CUBE + (x % UNIT_CUBE) / UNIT_CUBE, y // UNIT_CUBE + (y % UNIT_CUBE) / UNIT_CUBE,
                           colorseed[2])
            else:
                r, g, b = 1, 1, 1
            data[x, y] = list(map(int, (255 * r * value, 255 * g * value, 255 * b * value)))

    img = smp.toimage(data)
    img.show()
    img.save('noise.bmp')
    print("Time elapsed: " + str(time() - starttime))
    lrulist = [fade, hash_row, inc, grad, lerp, perlin]
    for lru in lrulist:
        try:
            print(lru.__name__, lru.cache_info())
        except AttributeError:
            continue


if __name__ == "__main__":
    main()
