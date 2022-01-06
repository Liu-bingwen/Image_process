from PIL import Image
import matplotlib.pyplot as plt #绘制图片进行显示
import numpy as np
import math

#放大缩小
def resize(f,sizew,sizeh):
    #得到需要的图的大小
    w=f.width
    h=f.height
    neww=int(w*sizew)
    newh=int(h*sizeh)
    #新建元组 生成新图像
    a=tuple([neww,newh])
    img=Image.new(f.mode,a,0)
    #变换为矩阵r和g 方便运算赋值
    r=np.array(f)
    g=np.array(img)
    for i in range(newh):
        for j in range(neww):
            x = int(i / sizeh)
            y = int(j / sizew)
            #使超出边界的x ,y为0
            if (x < 0):
                x = 0
            if (y < 0):
                y = 0
            if x < h and y < w:
                g[i, j] = r[x, y]
    g=Image.fromarray(g)
    return g


#实现图像的平移
def translate(f,x,y):
    height=f.height
    width=f.width
    #建立元组 生成新图像
    a=tuple([width,height])
    img=Image.new(f.mode,a,0)
    #变换为矩阵r和img 方对其运算赋值
    r=np.array(f)
    img=np.array(img)
    #平移模板
    T = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    #对原始图像的元组进行遍历
    for i in range(height):
        for j in range(width):
            #得到图像上的一点为i,j,1的列向量
            b=np.array([i,j,1]).transpose()
            #与模板矩阵相乘得结果 x y
            test=np.dot(T,b)
            #第一个值为x 第二个为y
            x=test[0]
            y=test[1]
            #判断是否在新图像矩阵中
            if x<height and y<width and x >= 0 and y>=0:
                img[x,y]=r[i,j]
    #array转换为image类型
    img=Image.fromarray(img)
    return img

#图像的旋转
def imrotate(f,angle):
    #得到高、宽、中心坐标
    height=f.height
    width=f.width

    centerx=int(height/2)
    centery=int(width/2)
    # 建立元组 生成新图像
    a = tuple([width, height])
    img = Image.new(f.mode, a, 0)
    # 变换为矩阵r和img 方对其运算赋值
    r = np.array(f)
    img = np.array(img)
    #旋转模板
    T=np.array([[math.cos(angle), -math.sin(angle), 0],[math.sin(angle), math.cos(angle), 0],[0, 0, 1]])

    for i in range(height):
        for j in range(width):
            #得到点到圆心得距离向量
            b=np.array([i-centerx,j-centery,1])
            #相乘
            test=np.dot(T,b)
            #得到x和y的值
            x=int(test[0])+centerx
            y=int(test[1])+centery
            # 判断是否在新图像矩阵中
            if x < height and y < width and x >= 0 and y >= 0:
                img[x, y] = r[i, j]
    # array转换为image类型
    img = Image.fromarray(img)
    return img

#图像翻转 x轴 y轴
def turnover(f,xory):
    # 得到高、宽、中心坐标
    height = f.height
    width = f.width
    # 建立元组 生成新图像
    a = tuple([width, height])
    img = Image.new(f.mode, a, 0)
    # 变换为矩阵r和img 方对其运算赋值
    r = np.array(f)
    img = np.array(img)
    #判断对x轴还是对y轴变换
    if xory=='x':
        for i in range(height):
            for j in range(width):
                img[i,j]=r[i][width-1-j]
    if xory=='y':
        for i in range(height):
            for j in range(width):
                img[i,j]=r[height-1-i][width]
    #将其转化为图像
    img=Image.fromarray(img)
    return img







if __name__ == '__main__':
    img=Image.open('fig0101.tif')
    g=resize(img,2,1)
    plt.figure("图像变换")
    plt.subplot(221)
    plt.imshow(g)
    plt.title('resize')

    g1=translate(img,30,90)
    plt.subplot(222)
    plt.imshow(g1)
    plt.title('translate')

    g2 = imrotate(img, 50)
    plt.subplot(223)
    plt.imshow(g2)
    plt.title('imrotate')


    g3=turnover(img,'x')
    plt.subplot(224)
    plt.imshow(g3)
    plt.title('turnover x')
    plt.show()
