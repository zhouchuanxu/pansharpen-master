from osgeo import gdal, gdalconst
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance

def result_to_image(img):#转换格式
    img = np.array(img)
    max_val, min_val = np.nanmax(img), np.nanmin(img)
    print(max_val, min_val)
    out = (img.astype(np.float) - min_val) / (max_val - min_val)#将像素值换成0~1的值
    out = out*255   #乘以255，像素值换成颜色值
    out = np.uint8(out)#utf-8编码格式转换
    return out
def PCA(rgb,b8):
    low=gdal.Open(rgb)
    hight=gdal.Open(b8)

    hight_band = hight.GetRasterBand(1)  # 获取全色图像的第一个波段
    hight_array = hight_band.ReadAsArray().astype(np.float32)  # 读取第一个波段的数据
    #hight_array=hight.ReadAsArray().astype(np.float32)# 数值化全色图像

    hight_x,hight_y=hight.RasterXSize,hight.RasterYSize#全色图像的维度
    R=low.GetRasterBand(1).ReadAsArray().astype(np.float32)# 数值化多光谱R波段
    G=low.GetRasterBand(2).ReadAsArray().astype(np.float32)# 数值化多光谱G波段
    B=low.GetRasterBand(3).ReadAsArray().astype(np.float32)# 数值化多光谱B波段

    ##将低分辨率的RGB影像重采样为全色波段的大小
    Rresize=cv2.resize(R,(hight_x,hight_y),interpolation=cv2.INTER_LINEAR)
    Gresize=cv2.resize(G,(hight_x,hight_y),interpolation=cv2.INTER_LINEAR)
    Bresize=cv2.resize(B,(hight_x,hight_y),interpolation=cv2.INTER_LINEAR)
    resample_RGB=np.array((Rresize,Gresize,Bresize)).astype(np.float32)
    #MS 将重采样的多光谱影像整合为RGB

    RGB_resample = resample_RGB.reshape((hight_x*hight_y, 3)) #MS
    #将这个三维数组重新形状为一个二维数组，其中每行包含一个像素的RGB通道值
    b8 = hight_array.reshape((hight_x*hight_y, 1))   #PAN调整为一列
    RGB_resample_mean = np.mean(RGB_resample, 0)
    # 按照第一维度(也就是括号里的0)求和，结果维度为（3，1） 第一个维度上（即各波段）的均值
    #RGB_resample数组沿着轴0（垂直方向）的均值。这将返回一个包含三个元素的一维数组，每个元素表示对应通道（红、绿、蓝）的均值。
    #就是每一列的均值，就是RGB三个颜色的均值
    RgbReduceMean = RGB_resample - np.tile(RGB_resample_mean, (hight_x*hight_y, 1))#去均值
    #将重采样后的多光谱图像减去均值，实现去均值操作。每一列都减去均值

    covar = (np.matrix(RgbReduceMean).T * np.matrix(RgbReduceMean)) / (hight_x*hight_y)#求协方差
    #是将RgbReduceMean数组转换为一个列向量，.T 表示进行转置操作，将其转换为一个行向量。
    #np.matrix(RgbReduceMean) 表示将RgbReduceMean数组转换为一个矩阵。
    #将转置后的行向量与原始矩阵相乘，得到一个矩阵。然后将该矩阵除以图像中的像素总数，以计算协方差矩阵。
    #计算去均值后的多光谱图像的协方差矩阵。

    value, vector = np.linalg.eig(covar)  # 计算协方差矩阵特征向量
    #np.linalg.eig(covar) 表示对协方差矩阵进行特征值和特征向量的计算。
    #value 是一个包含协方差矩阵的特征值的一维数组。
    #vector 是一个二维数组，其中每列是协方差矩阵的特征向量。
    vector = np.fliplr(vector)  # 左右对调特征向量

    rgb2pc = np.array(np.matrix(RGB_resample) * np.matrix(vector))
    # PCA正变换  将去均值后的多光谱图像通过特征向量进行线性变换，得到主成分分析后的结果

    rgb2pc[:][0] = b8[:][0]  #PAN调整为一列的值给矩阵运算后的
    #第一主分量用全色波段PAN代替 将主成分分析后的结果中的第一主分量用全色图像的第一个波段代替。


    RGB_resample = np.array(np.matrix(rgb2pc) * np.linalg.inv(vector)).reshape(3,hight_y, hight_x)#pca逆变换并转换为三维矩阵
    #将经过主成分分析后的结果通过特征向量的逆矩阵进行线性变换，恢复到原始的多光谱波段空间
    #np.linalg.inv(vector) 计算特征向量矩阵 vector 的逆矩阵。
    result = np.uint8(RGB_resample)
    img1 = cv2.merge([result[0][:][:],result[1][:][:],result[2][:][:]])
    #将融合后的三个波段重新合并成一幅图像

    img=Image.fromarray(img1)   
    img_light=img
    img_light.show()
    img_light.save(r"D:\allproject\pansharpen\pansharpen-master\PCAt.png","png")
    #cv2.imwrite(r"D:\CDUT\intership\experiment1\PCA2.png", img1)
def main():
    original_rgb = r"D:\allproject\pansharpen\mst.png"
    original_b8 = r"D:\allproject\pansharpen\pant.png"
    #original_rgb = r"D:\allproject\pansharpen\pansharpen-master\index.tif"
    #original_b8 = r"D:\allproject\pansharpen\pansharpen-master\b8.tif"
    PCA(original_rgb,original_b8)
if __name__ == '__main__':
    main()
    