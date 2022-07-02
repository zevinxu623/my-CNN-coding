import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import os
import imageio

def Dcm2jpg(file_path):
    # 获取所有图片名称
    c = []
    names = os.listdir(file_path)  # 路径

    # 将文件夹中的文件名称与后边的 .dcm分开
    for name in names:
        index = name.rfind('.')
        name = name[:index]
        c.append(name)

    for files in c:
        picture_path = "/home/xuzhiwen/enhance004/" + files + ".dcm"
        out_path = "/home/xuzhiwen/new/enhance004/" + files + ".jpg"
        ds = pydicom.read_file(picture_path)
        img = ds.pixel_array  # 提取图像信息
        imageio.imsave(out_path, np.uint8(img/10))

    print('all is changed')


if __name__ == '__main__':
    Dcm2jpg('/home/xuzhiwen/enhance004')
