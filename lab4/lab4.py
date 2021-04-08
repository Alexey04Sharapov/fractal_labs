import cv2
import numpy as np

def get_base(image, function, coeff):
    k, l = image.shape
    u = np.ndarray(image.shape)

    for i in range(k):
        for j in range(l):
            minmax = []

            if i > 0:
                minmax.append(image[i - 1][j])
            if i < image.shape[0] - 1:
                minmax.append(image[i + 1][j])
            if j < image.shape[1] - 1:
                minmax.append(image[i][j + 1])
            if j > 0:
                minmax.append(image[i][j - 1])
            u[i][j] = function(image[i][j] + coeff, function(minmax))
    return u

def get_vol(u, b):
    k, l = u.shape
    vol = 0.0
    for i in range(k):
        for j in range(l):
            vol += u[i][j] - b[i][j]
    return vol

def get_d(image):
    u1 = get_base(image, max, 1)
    u2 = get_base(u1, max, 1)

    b1 = get_base(image, min, -1)
    b2 = get_base(b1, min, -1)

    vol1 = get_vol(u1, b1)
    vol2 = get_vol(u2, b2)

    A_1 = (vol1 - 0) / 2.0
    A_2 = (vol2 - vol1) / 2.0

    coeff = (np.log(A_1) - np.log(A_2)) / (np.log(1) - np.log(2)) + 1

    return 2 - coeff

if __name__ == '__main__':
    image = cv2.imread('../images/img_res_3.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res = get_d(image)
    print(res)
