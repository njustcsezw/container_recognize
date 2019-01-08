import cv2
import numpy as np
import container_recognize.picture_hough_transform as hgt


#find father and update
def find_fa(x):
    global count, fa, cc

    fx = fa[x]
    if fa[fx] == fx:
        #if father has no father, no more search
        return fx
    else:
        #update x's father
        fa[x] = find_fa(fx)
        return fa[x]


def two_pass(binary_img, mask=255, area=10):
    #init merge and find set
    global count, fa, cc

    fa = list(range(binary_img.shape[0] * binary_img.shape[1]))
    #father node
    cc = np.zeros(binary_img.shape[0] * binary_img.shape[1])
    #count connected components area of fa[]
    cc = cc+1

    dx = [0, 0, -1, 1, -1, -1, 1, 1]
    dy = [-1, 1, 0, 0, -1, 1, -1, 1]

    for i in list(range(binary_img.shape[0])):
        for j in list(range(binary_img.shape[1])):
            if binary_img[i, j] == mask:
                for DIR in list(range(8)):
                    nx = dx[DIR] + i
                    ny = dy[DIR] + j
                    if nx >= 0:
                        if nx < binary_img.shape[0]:
                            if ny >= 0:
                                if ny < binary_img.shape[1]:
                                    if binary_img[nx, ny] == mask:
                                        a = i * binary_img.shape[1] + j
                                        b = nx * binary_img.shape[1] + ny
                                        pa = find_fa(a)
                                        #shorten chain
                                        pb = find_fa(b)
                                        #merge father
                                        if pa < pb:
                                            fa[pb] = pa
                                            cc[pa] += cc[pb]
                                            cc[pb] = 0
                                        elif pa > pb:
                                            fa[pa] = pb
                                            cc[pb] += cc[pa]
                                            cc[pa] = 0

    #for i in list(range(binary_img.shape[0])):
        #for j in list(range(binary_img.shape[1])):
            #if binary_img[i, j] == mask:
                #a = i * binary_img.shape[1] + j
                #find_fa(a)

    count = 0
    colormap = np.zeros((binary_img.shape[0], binary_img.shape[1], 3))
    #color hash table
    for i in list(range(binary_img.shape[0])):
        for j in list(range(binary_img.shape[1])):
            if binary_img[i, j] == mask:
                a = i * binary_img.shape[1] + j
                pa = find_fa(a)
                if cc[pa] >= area:
                    # connected components with area >= 100 pixels
                    pa_i = int(pa / binary_img.shape[1])
                    pa_j = int(pa % binary_img.shape[1])
                    if np.max(colormap[pa_i, pa_j, :]) == 0:
                        colormap[pa_i, pa_j, :] = np.random.randint(256, size=3)
                        count += 1
                    colormap[i, j, :] = colormap[pa_i, pa_j, :]
    print(count)
    return colormap

img_path = r'pictures/a6.jpg'
img = cv2.imread(img_path)

#转换灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#霍夫变换
hgt = hgt.picture_hough_transform(gray)
(_, thresh1) = cv2.threshold(hgt, 150, 255, cv2.THRESH_BINARY_INV)
#cv2.imwrite(r'pictures/hougu.jpg', thresh1)
des = two_pass(thresh1, 255, 10)


cv2.imshow('hough_img', hgt)
#cv2.imshow('hough_img2', thresh1)
cv2.imshow('hough_img1', des)
cv2.waitKey(0)
#cv2.imwrite(save_path, crop_img)
cv2.destroyAllWindows()


