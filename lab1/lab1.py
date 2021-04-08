import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu

def boxcounting(bi, k):
    a = np.add.reduceat(bi, np.arange(0, bi.shape[0], k), axis = 0)
    b = np.arange(0, bi.shape[1], k)

    s = np.add.reduceat(a, b, axis = 1)

    return len(np.where((s > 0) & (s < k ** 2))[0])


def cap_dim(image):
    bi = image < threshold_otsu(image)
    t = 2 ** np.floor(np.log(min(bi.shape)) / np.log(2))
    n = int(np.log(t) / np.log(2))

    sizes = 2 ** np.arange(n, 0, -1)
    counts = []

    for size in sizes:
        counts.append(boxcounting(bi, size))

    cArr = np.polyfit(np.log(sizes), np.log(counts), 1)
    return cArr[0]

if __name__ == '__main__':
    image = io.imread('../images/img_res_3.jpg')
    image = color.rgb2gray(image)

    res = -cap_dim(image)
    print(res)