import matplotlib

if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    import random
    import numpy as np


    pic = imageio.imread('dssa.jpg')

    low_pixel = pic < 50

    # to ensure of it let's check if all values in low_pixel are True or not
    if low_pixel.any() == True:
        print(low_pixel.shape)
    print(pic[low_pixel])
    pic[low_pixel] = random.randint(25,255)

    print(pic.shape)
    print(low_pixel.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(pic)
    plt.show()