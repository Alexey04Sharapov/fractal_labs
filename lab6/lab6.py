import cv2
import numpy as np
import matplotlib.pyplot as plot

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

def get_A(image, delta):
    u_arr = [image]
    b_arr = [image]

    u_arr.append(get_base(u_arr[0], max, 1))
    b_arr.append(get_base(u_arr[0], min, -1))

    a = (get_vol(u_arr[-1], b_arr[-1]) - get_vol(u_arr[-2], b_arr[-2])) / 2.0
    yield np.log(a) / np.log(delta)

if __name__ == '__main__':
    image = cv2.imread('../images/img_res_3.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    delta = [i for i in range (2, 11)]
    a = [i for i in get_A(image, delta)]

    plot.figure(figsize = (10, 10))
    plot.xlabel('d')
    plot.ylabel('log(A) / log(d)')
    plot.grid(True)

    line, = plot.plot(delta, a[0])
    
    lines = []
    lines.append(line)

    plot.legend(lines)
    plot.savefig('../images/img_res_6.jpg')
