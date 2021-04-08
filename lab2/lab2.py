from skimage import io

RED = 0
GREEN = 1
BLUE = 2


if __name__ == '__main__':
    image = io.imread('../images/img.jpg')
    image[:, :, [RED, BLUE]] = 0.0 #Save only green color
    
    io.imsave('../images/img_res_2.jpg', image, check_contrast = False)