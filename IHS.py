from osgeo import gdal, gdalconst

import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
def IHS(rgb,b8):

    low=gdal.Open(rgb)  #MS 有颜色
    hight=gdal.Open(b8) #PAN 黑白

    hight_band = hight.GetRasterBand(1)  # PANk,ik获取全色图像的第一个波段
    hight_array = hight_band.ReadAsArray().astype(np.float32)  # 读取第一个波段的数据
    #hight_array = hight.ReadAsArray().astype(np.float32)# 数值化全色图像 PAN

    hight_x,hight_y = hight.RasterXSize,hight.RasterYSize#PAN全色图像的维度
    #，可以获取全色图像数据集的维度信息。这些信息对于图像处理和分析非常重要，因为它们可以用于确定图像的大小、
    # 定义处理操作的边界和范围，以及进行像素级的操作和计算

    R=low.GetRasterBand(1).ReadAsArray().astype(np.float)# 数值化多光谱R波段 MS
    G=low.GetRasterBand(2).ReadAsArray().astype(np.float)# 数值化多光谱G波段 MS
    B=low.GetRasterBand(3).ReadAsArray().astype(np.float)# 数值化多光谱B波段



    Rresize=cv2.resize(R,(hight_x,hight_y),interpolation=cv2.INTER_NEAREST)#将低分辨率的RGB影像重采样为全色波段的大小
    Gresize=cv2.resize(G,(hight_x,hight_y),interpolation=cv2.INTER_NEAREST)
    Bresize=cv2.resize(B,(hight_x,hight_y),interpolation=cv2.INTER_NEAREST)
    #interpolation=cv2.INTER_NEAREST 是插值方法的选择。在这里，cv2.INTER_NEAREST 采用最近邻插值方法，即对于目标图像中的每个像素，从输入图像中选择最接近的像素值作为其值
    #将低分辨率的RGB影像重采样为与全色波段相同大小的图像。具体来说，它会将红、绿、蓝三个波段的图像（R、G、B）分别进行重采样，
    # 使它们的大小与全色波段图像（hight_x和hight_y表示的尺寸）一致。

    resample_RGB=np.array((Rresize,Gresize,Bresize)).astype(np.float32)
    #MS 将重采样的多光谱影像整合为RGB

    reshapeR=Rresize.reshape(1,hight_x*hight_y)
    reshapeG=Gresize.reshape(1,hight_x*hight_y)
    reshapeB=Bresize.reshape(1,hight_x*hight_y)

    change=np.concatenate((reshapeR,reshapeG,reshapeB),axis=0) #MS图像变成一维的

    #reshapeb8=hight_array.reshape(1,hight_x*hight_y*3)  #PAN图像
    print(hight_array.shape)
    print(hight_x)
    print(hight_y)
    reshapeb8 = hight_array.reshape(1, hight_x * hight_y)  # PAN图像
    #将重采样后的RGB波段转换为1D形式，方便后续计算。
    #这些操作的目的是将重采样后的红、绿、蓝通道图像数据和原始全色图像数据进行对齐，并将它们转换为一维数组的形式。
    # 这样做可以方便后续的图像处理和计算操作

    tran1 = np.array([[1/3,1/3,1/3],
            [-np.sqrt(2)/6,-np.sqrt(2)/6,2*np.sqrt(2)/6],
            [1/np.sqrt(2),-1/np.sqrt(2),0]])#正变换矩阵
    #tran1 是一个 3x3 的矩阵，用于将 RGB 图像转换为 YIQ 颜色空间。YIQ 是一种表示颜色的方式，其中 Y 通道表示亮度信息，而 I 和 Q 通道表示色度信息。
    # 通过将 RGB 图像与 tran1 矩阵相乘，可以得到对应的 YIQ 颜色空间的图像。RGB->IHS
    tran2= np.array([[1,-1/np.sqrt(2),1/np.sqrt(2)],
          [1,-1/np.sqrt(2),-1/np.sqrt(2)],
          [1,np.sqrt(2),0]])#逆变换矩阵
    #tran2 是一个 2x3 的矩阵，用于将 YIQ 图像的 I 和 Q 通道进行压缩变换。这种变换被称为色度压缩，它可以减少色彩信息的冗余，从而实现图像的压缩和存储优化。
    # 通过将 YIQ 图像的 I 和 Q 通道与 tran2 矩阵相乘，可以得到压缩后的 I' 和 Q' 通道。
    #定义正变换矩阵tran1和逆变换矩阵tran2 IHS->RGB


    #RGB--->IHS
    rgb2ihs=np.dot(tran1,change)  #MS图像变成IHS

    rgb2ihs[0,:]=reshapeb8  #PAN变成一维图像了 替代MS变成IHS的第一个通道
    #第一行（rgb2ihs[0, :]）将被替换为之前重塑的全色图像数据 reshapeb8

    ihs2rgb=np.dot(tran2,rgb2ihs)
    #利用正变换矩阵将RGB图像转换为IHS空间，然后将全色图像的强度（I）通道替换为低分辨率的全色图像，再利用逆变换矩阵将IHS空间转换回RGB空间。

    p=np.zeros(((3,hight_x,hight_y)))
    p[0,:,:]=ihs2rgb[0].reshape((hight_x,hight_y))
    #将转换回 RGB 颜色空间后的第一个通道（亮度）的数据重新塑形为 (hight_x, hight_y) 的二维数组
    p[1,:,:]=ihs2rgb[1].reshape((hight_x,hight_y))
    #色调
    p[2,:,:]=ihs2rgb[2].reshape((hight_x,hight_y))
    #饱和度
    #这样做的目的是将转换回 RGB 颜色空间的图像数据重新组织成与原始图像相同的形状和尺寸。
    # 通过重新塑形并赋值给 p，可以将转换后的图像数据存储在 p 中，

    #创建一个全零数组p，大小为(3, hight_x, hight_y)，然后将转换后的RGB通道重新赋值给p。
    #p=np.uint8(p)
    img1 = cv2.merge([p[2],p[1],p[0]])

    cv2.imwrite(r"D:\allproject\pansharpen\pansharpen-master\IHS2-urban.png", img1)
    #使用cv2.merge函数将通道顺序改为BGR，然后保存图像为PNG格式

def main():
    original_rgb = r"D:\allproject\pansharpen\cutgroups\2-urban\I_MS.tif"
    original_b8 = r"D:\allproject\pansharpen\cutgroups\2-urban\I_PAN.tif"
    #original_rgb = r"D:\allproject\pansharpen\pansharpen-master\index.tif"
    #original_b8 = r"D:\allproject\pansharpen\pansharpen-master\b8.tif"
    IHS(original_rgb,original_b8)
if __name__ == '__main__':
    main()
    