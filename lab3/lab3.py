from skimage import io, color

if __name__ == '__main__':
    image = io.imread('../images/img.jpg')
    image = color.rgb2gray(image)
    
    io.imsave('../images/img_res_3.jpg', image, check_contrast = False)