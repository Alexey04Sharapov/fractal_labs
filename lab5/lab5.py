import cv2
import numpy as np

CELL_SIZE = 10
A_THERSHOLD = 100

def get_vol(u, b):
    k, l = u.shape
    vol = 0.0
    for i in range(k):
        for j in range(l):
            vol += u[i][j] - b[i][j]
    return vol

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

def get_A(image):
    u1 = get_base(image, max, 1)
    u2 = get_base(u1, max, 1)

    b1 = get_base(image, min, -1)
    b2 = get_base(b1, min, -1)

    return (get_vol(u2, b2) - get_vol(u1, b1)) / 2.0

if __name__ == '__main__':
    image = cv2.imread('../images/img_5.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    si = np.full(image.shape, 255)

    A_n = []
    for i in range(0, image.shape[0], CELL_SIZE):
        for j in range(0, image.shape[1], CELL_SIZE):
            A = get_A(image[i:i + CELL_SIZE, j: j + CELL_SIZE])
            A_n.append(A)

            if A > A_THERSHOLD:
                si[i:i + CELL_SIZE,
                j:j + CELL_SIZE].fill(0)

    cv2.imwrite('../images/img_res_5.jpg', si)
