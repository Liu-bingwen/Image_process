from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from  scipy import interpolate
import pylab as pl
from scipy import signal,misc
import cv2 as cv
import matplotlib.image as mpimg
import math

###图片的打开、显示与保存
def Image_Open():
    img = Image.open("D:/桌面/数字图像处理/课程设计1/airport.tif")
    img.show()
    img.save('D:/airport.tif')


###图像的几何变换
class Image_Geo:
    def __init__(self,image):
        self.img1 = image


    ###图像的放大
    def Image_Bigger(self,times):
        image = np.array(img1)
        [h,w,c] = image.shape
        #临近插值
        img_new = np.zeros([times*h,times*w,c],np.float32)
        img_new = np.zeros([times * h, times * w, c], np.float32)
        for i in range(c):
            for j in range(h):
                for k in range(w):
                    img_new[int(times * j - times / 2):int(times * j + times / 2),
                    int(times * k - times / 2):int(times * k + times / 2), i] = img1[j, k, i]

        img_new2 = Image.fromarray(np.uint8(img_new))
        img_new2.show()

###图像的缩小
    def Image_smaller(self,times):
        image = np.array(img1)
        [h, w,c] = image.shape
        img_new = np.zeros([int(h / times), int(w / times),c], np.float32)
        for i in range(c):
            for j in range(int(h / times)):
                for k in range(int(w / times)):
                    img_new[j, k, i] = image[j * times, k * times, i]
        img_new2 = Image.fromarray(np.uint8(img_new))
        img_new2.show()

    def Image_move(self,x,y):
        h = img1.height
        w = img1.width
        img_arr = np.array(img1)
        img_new = np.zeros(img_arr.shape,np.uint8)
        image = np.array(img1)
        img_new = np.array(img_new)
        T = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

        for i in range(h):
            for j in range(w):
                b = np.array([i, j, 1]).transpose()
                test = np.dot(T, b)
                x = test[0]
                y = test[1]
                if x < h and y < w and x >= 0 and y >= 0:
                    img_new[x, y] = image[i, j]

        img_new2 = Image.fromarray(img_new)

        # self.img1.show()
        # img_new2.show()
        plt.figure('图像平移')
        plt.subplot(121)
        plt.imshow(img1,cmap='gray')
        plt.title('origin')

        plt.subplot(122)
        plt.imshow(img_new2,cmap = 'gray')
        plt.title('after')
        plt.show()

        #pass

    ###图像的旋转
    def Image_rotate(self,angle):
        h = img1.height
        w = img1.width
        img_arr = np.array(img1)
        centerx = int(h/2)
        centery = int(w/2)
        img_new = np.zeros(img_arr.shape,np.uint8) #Image.new(img1.mode,a,0)
        image = np.array(img1)
        img_new = np.array(img_new)
        T = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

        for i in range(h):
            for j in range(w):
                b=np.array([i-centerx,j-centery,1])
                test=np.dot(T,b)
                x = int(test[0]) + centerx
                y = int(test[1]) + centery
                if x < h and y < w and x >= 0 and y>=0:
                    img_new[x,y] = image[i,j]

        img_new2 = Image.fromarray(img_new)
        img_new2.show()


        #pass

    def Image_flip(self,dir):
        h = img1.height
        w = img1.width
        img_arr = np.array(img1)
        img_new = np.zeros(img_arr.shape,np.uint8)
        image = np.array(img1)
        img_new = np.array(img_new)
        if dir == 'x':
            for i in range(h):
                for j in range(w):
                    img_new[i, j] = image[i][w - 1 - j]

        if dir == 'y':
            for i in range(h):
                for j in range(w):
                    img_new[i, j] = image[h - 1 - i][w]

        img_new2 = Image.fromarray(img_new)
        img_new2.show()

        #pass

    

###图像的像素变换

class Image_Pixel:
    def __init__(self,image1,image2):
        self.img1 = image1
        self.img2 = image2

    ###图像的合成
    def Image_composite(self):
        width = min(self.img1.size[0],self.img2.size[0])
        height = min(self.img1.size[1],self.img2.size[1])
        img_new = Image.new('RGB',(width,height))
        for x in range(width):
            for y in range(height):
                if y % 2 == 0:
                    pixel = self.img1.getpixel((x, y))
                    img_new.putpixel((x, y), pixel)
                else:
                    pixel = self.img2.getpixel((x, y))
                    img_new.putpixel((x, y), pixel)
        img_new2 = Image.fromarray(np.uint8(img_new))
        img_new2.show()
        #pass

    ###图像转变成灰度图
    def Image_graytrans(self):
        img = np.array(self.img1)
        R,G,B = img[:,:,0],img[:,:,1],img[:,:,2]
        img_gray = 0.2989 * R + 0.587* G + 0.1140 * B
        plt.imshow(img_gray,cmap='gray')
        plt.show()
        #pass

    ###直方图的计算与变换
    def histcal(self):
        img = np.array(self.img1)
        scr = np.array(self.img2)
        mHist1 = []
        mNum1 = []
        inhist1 = []
        mHist2 = []
        mNum2 = []
        inhist2 = []
        # 对原图像进行均衡化
        for i in range(256):
            mHist1.append(0)
        row, col = img.shape  # 获取原图像像素点的宽度和高度
        for i in range(row):
            for j in range(col):
                mHist1[img[i, j]] = mHist1[img[i, j]] + 1  # 统计灰度值的个数
        mNum1.append(mHist1[0] / img.size)
        for i in range(0, 255):
            mNum1.append(mNum1[i] + mHist1[i + 1] / img.size)
        for i in range(256):
            inhist1.append(round(255 * mNum1[i]))
        # 对目标图像进行均衡化
        for i in range(256):
            mHist2.append(0)
        rows, cols = scr.shape  # 获取目标图像像素点的宽度和高度
        for i in range(rows):
            for j in range(cols):
                mHist2[scr[i, j]] = mHist2[scr[i, j]] + 1  # 统计灰度值的个数
        mNum2.append(mHist2[0] / scr.size)
        for i in range(0, 255):
            mNum2.append(mNum2[i] + mHist2[i + 1] / scr.size)
        for i in range(256):
            inhist2.append(round(255 * mNum2[i]))

        # 进行规定化
        g = []  # 用于放入规定化后的图片像素
        for i in range(256):
            a = inhist1[i]
            flag = True
            for j in range(256):
                if inhist2[j] == a:
                    g.append(j)
                    flag = False
                    break
            if flag == True:
                minp = 255
                for j in range(256):
                    b = abs(inhist2[j] - a)
                    if b < minp:
                        minp = b
                        jmin = j
                g.append(jmin)

        for i in range(row):
            for j in range(col):
                img[i, j] = g[img[i, j]]

        #self.img1.show()
        img_new = Image.fromarray(img)
        img_new.show()
        plt.hist(img.ravel(),256)
        plt.show()

class denoise:
    def __init__(self,image1):
        self.img1 = image1

    def medi_filter(self):
        image = np.array(img1)
        [h,w,c] = image.shape
        k=7
        pad = k//2
        img_new = np.zeros((h + 2* pad,w+2*pad,c),dtype=np.float)
        img_new[pad:pad+h,pad:pad+w] = image.copy().astype(np.float)

        temp = img_new.copy()
        for y in range(h):
            for x in range(w):
                for ci in range(c):
                    img_new[pad+y,pad+x,ci] = np.median(temp[y:y+k,x:x+k,ci])

        img_new = img_new[pad:pad+h,pad:pad+w].astype(np.uint8)

        img_new2 = Image.fromarray(img_new)
        img_new2.show()




        #pass

    def mean_filter(self):
        pass

    def spatial_filter(self):
        pass

    def freq_filter(self):
        pass


###锐化与边缘检测
class sharp_edge:
    def __init__(self,image1,image2):
        self.img1 = image1
        self.img2 = image2

    ###sobel算子对图像对图像进行边缘检测
    def edge(self):
        image = np.array(img1)
        h = image.shape[0]
        w = image.shape[1]
        image_new = np.zeros(image.shape,np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                sx = (image[i + 1][j - 1] + 2 * image[i + 1][j] + image[i + 1][j + 1]) - \
                     (image[i - 1][j - 1] + 2 * image[i - 1][j] + image[i - 1][j + 1])
                sy = (image[i - 1][j + 1] + 2 * image[i][j + 1] + image[i + 1][j + 1]) - \
                     (image[i - 1][j - 1] + 2 * image[i][j - 1] + image[i + 1][j - 1])
                image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))

        img_new2 = Image.fromarray(image_new)
        img_new2.show()


class segmentation:
    def __init__(self,image1,image2):
        self.img1 = image1
        self.img2 = image2

    def calhist(self):
        image = self.img1
        image = np.array(img1)
        rows,cols = image.shape
        grayHist = np.zeros([256], np.uint64)
        for r in range(rows):
            for c in range(cols):
                grayHist[image[r][c]] += 1
        return grayHist

    ###利用直方图得到阈值并进行分割（灰度图）
    def threshold(self):
        histogram = self.calhist()
        maxLoc = np.where(histogram == np.max(histogram))
        firstPeak = maxLoc[0][0]
        measureDists = np.zeros([256], np.float32)
        for k in range(256):
            measureDists[k] = pow(k - firstPeak, 2) * histogram[k]
        maxLoc2 = np.where(measureDists == np.max(measureDists))
        secondPeak = maxLoc2[0][0]
        thresh = 0
        if firstPeak > secondPeak:
            temp = histogram[int(secondPeak): int(firstPeak)]
            minLoc = np.where(temp == np.min(temp))
            thresh = secondPeak + minLoc[0][0] + 1
        else:
            temp = histogram[int(firstPeak): int(secondPeak)]
            minLoc = np.where(temp == np.min(temp))
            thresh = firstPeak + minLoc[0][0] + 1

        # 找到阈值，我们进行处理
        #img = image.copy()
        image = np.array(self.img1)
        image[image > thresh] = 255
        image[image <= thresh] = 0

        img_new = Image.fromarray(image)
        img_new.show()






if __name__ == '__main__':
    #img1 = Image.open("D:/桌面/数字图像处理/课程设计1/lenna_RGB.tif")
    #img2 = Image.open("D:/桌面/数字图像处理/课程设计1/target.png")
    #img1 = Image.open('D:/桌面/数字图像处理/课程设计1/airport.tif')
    img1 = Image.open('D:/桌面/数字图像处理/课程设计1/noise.tif')
    #img1 = mpimg.imread('D:/桌面/数字图像处理/课程设计1/lenna_RGB.tif')
    #img2 = np.array(img2)
    #img1 = np.array(img1)
    # img1 = cv.imread("D:/airport.tif")
    #rows = img1.shape[0]
    #cols = img1.shape[1]

    ###图像的几何变换
    #img = Image_Geo(img1)
    #img.Image_Bigger(2)       #图像的放大
    #img.Image_smaller(2)      #图像的缩小
    #img.Image_move(20,20)     #图像的平移
    #img.Image_rotate(40)      #图像的旋转
    #img.Image_flip('x')        #图像的反转

    ###图像的去噪
    img = denoise(img1)
    img.medi_filter()          #中值滤波

    ###图像的像素变换
    #img = Image_Pixel(img1,img2)
    #img.Image_composite()     #图像的合成
    #img.Image_graytrans()     #图像转变成灰度图
    #img.histcal()             #直方图均衡化

    ###图像锐化与边缘检测
    #img = sharp_edge(img1,img2)
    #img.edge()                #图像的边缘检测

    ###图像的分割
    #img = segmentation(img1,img2)
    #img.threshold()









