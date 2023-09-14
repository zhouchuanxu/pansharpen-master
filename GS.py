from osgeo import gdal, gdalconst
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
def GS(rgb,b8):
    low=gdal.Open(rgb)  #MS
    hight=gdal.Open(b8)

    hight_array=hight.ReadAsArray().astype(np.float32)# 数值化全色图像
    hight_x,hight_y=hight.RasterXSize,hight.RasterYSize#全色图像的维度

    R=low.GetRasterBand(1).ReadAsArray().astype(np.float)# 数值化多光谱R波段
    G=low.GetRasterBand(2).ReadAsArray().astype(np.float)# 数值化多光谱G波段
    B=low.GetRasterBand(3).ReadAsArray().astype(np.float)# 数值化多光谱B波段
    Rresize=cv2.resize(R,(hight_x,hight_y),interpolation=cv2.INTER_LINEAR)#将低分辨率的RGB影像重采样为全色波段的大小
    Gresize=cv2.resize(G,(hight_x,hight_y),interpolation=cv2.INTER_LINEAR)
    Bresize=cv2.resize(B,(hight_x,hight_y),interpolation=cv2.INTER_LINEAR)
    resample_RGB=np.array((Rresize,Gresize,Bresize)).astype(np.float32)#将重采样的多光谱影像整合为RGB
    resample_RGB=cv2.merge([Rresize,Gresize,Bresize])

    #MS处理
    means = np.mean(resample_RGB, axis=(0, 1))
    #这一行计算了resample_RGB数组在指定轴（0和1）上的均值。它计算了每个颜色通道（R、G、B）的均值。
    image_lr = resample_RGB-means
    #这一行将resample_RGB减去均值，得到了中心化后的图像。
    #sintetic intensity
    I = np.mean(resample_RGB, axis=2, keepdims=True)
    #这一行计算了resample_RGB数组沿第二个轴（颜色通道轴）的均值，保持维度不变。从而得到了图像的灰度强度。
    I0 = I-np.mean(I)

    #PAN处理
    #这一行根据高度数组对灰度强度图像进行了归一化处理，使其具有与高度数组相似的范围和均值。
    image_hr = (hight_array-np.mean(hight_array))*(np.std(I0, ddof=1)/np.std(hight_array, ddof=1))+np.mean(I0)
    image_hr=np.reshape(image_hr,[hight_x,hight_y,1])
    #computing coefficients

    g = []
    g.append(1)

    for i in range(3): #这一行开始一个循环，遍历三个颜色通道
        temp_h = image_lr[:, :, i] #一行从中心化后的图像中提取当前颜色通道的值。
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        #这一行计算了灰度强度图像和当前颜色通道之间的协方差，并将结果存储在变量c中。
        g.append(c[0,1]/np.var(I0)) #这一行将协方差的归一化值作为系数添加到系数列表g中。

    g = np.array(g)
    print(I0.shape)
    #detail extraction
    delta = image_hr-I0  #PAN-MS的标准值
    
    deltam = np.tile(delta, (1, 1, 3+1))
    #这一行使用np.tile函数将delta数组沿第三个轴（颜色通道轴）复制为一个4维数组，其中每个颜色通道都有一个差异值。


    #fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    #这一行将中心化的灰度强度图像I0和中心化后的图像image_lr沿最后一个轴（颜色通道轴）连接起来，形成一个包含灰度强度和颜色信息的数组。
    #MS + PAN

    g = np.expand_dims(g, 0)
    #这一行在第0个维度上添加一个新的维度，将系数数组g转换为1维。
    g = np.expand_dims(g, 0) 
    g = np.tile(g, (hight_x, hight_y, 1))
    #这一行使用np.tile函数将系数数组g沿前两个维度复制为与V相同的形状，以便与差异数组deltam相乘。

    V_hat = V+ g*deltam
    #这一行对V和deltam进行逐元素相乘，并将结果与V相加，得到调整后的数组V_hat

    I_GS = V_hat[:, :, 1:]
    #这一行从调整后的数组V_hat中提取出颜色信息，即去除灰度强度信息
    I_GS = I_GS - np.mean(I_GS, axis=(0, 1))+means
    #这一行对提取出的颜色信息进行中心化，然后加上之前计算的均值means，得到最终的调整后的颜色信息。

    #adjustment
    change_rgb=I_GS#np.uint8(I_GS)
    print(change_rgb.shape)

    img1 = cv2.merge([change_rgb[:,:,2],change_rgb[:,:,1],change_rgb[:,:,0]])
    print(img1)
    #img=Image.fromarray(img1)
    #enhance = image_enhance(img, 1, 1.5) #调用类
    #img_light = enhance.image_brightened()
    #img.show()
   #img.save(r"D:\CDUT\intership\experiment1\GS.png","png")
    cv2.imwrite(r"D:\allproject\pansharpen\pansharpen-master\GS.png", img1)
def main():
    original_rgb = r"D:\allproject\pansharpen\pansharpen-master\index.tif"
    original_b8 = r"D:\allproject\pansharpen\pansharpen-master\b8.tif"
    GS(original_rgb,original_b8)
if __name__ == '__main__':
    main()