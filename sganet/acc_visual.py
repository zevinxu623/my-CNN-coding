# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/11/1 16:57
# @File     : acc_visual.py
# @Software : PyCharm

import os
import matplotlib.pyplot as plt
def visual_acc_from_file():
    # if not os.path.exists(file_path):
    #     assert "not find {0}!".format(file_path)
    file_path1 = "E:\deeplearningwork\my-CNN-coding\\inceptionV3\metric\inceptionV3-our-3.txt"
    file_path2 = "E:\deeplearningwork\my-CNN-coding\\inceptionV3\metric\inceptionV3-our-3.txt"
    file_path1 = "E:\deeplearningwork\my-CNN-coding\\resNet\metric\\resNet-our-1.txt"
    file_path2 = "E:\deeplearningwork\my-CNN-coding\\resNet\metric\\resNet-our-1.txt"
    # file_path3 = "models/4.21_Unet/HCM_fashion_weights/Adam0.001_batch64_L1/log.txt"
    with open(file_path1, "r", encoding='GBK') as f:
        datas1 = [list(map(str.strip, i.split(","))) for i in filter(lambda x: "test" in x, f.readlines())]
    with open(file_path2, "r") as f:
        datas2 = [list(map(str.strip, i.split(","))) for i in filter(lambda x: "train" in x, f.readlines())]
    # with open(file_path3, "r") as f:
    #     datas3 = [list(map(str.strip, i.split(","))) for i in filter(lambda x: "eval" in x, f.readlines())]

    acc1 = [float(i[1].split(" ")[-1]) for i in datas1]
    acc2 = [float(i[1].split(" ")[-1]) for i in datas2]
    #print(acc)
    # MSED_cc = [float(i[1].split(" ")[1]) for i in datas2]
    # Unet_cc = [float(i[1].split(" ")[1]) for i in datas3]
    # print(len(total_loss))
    plt.plot([i for i in range(len(acc1))],acc1, linewidth = 0.5, color ='b', label = "Testing")
    plt.plot([i for i in range(len(acc2))],acc2, linewidth = 0.5, color ='r', label = "Training")
    # plt.plot([i for i in range(len(MSED_cc))], MSED_cc, linewidth = 0.5, color='g', label="Pak et al. [8]")
    # plt.plot([i for i in range(len(MSED_cc))], Unet_cc, linewidth = 0.5, color='r', label="H. N. Abdullah et al. [4]")
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.title('Average Accuracy')
    plt.legend()

    #plt.show()
    plt.savefig("E:\deeplearningwork\my-CNN-coding\\resNet\metric\\acc-resNet-our-1.jpg", dpi = 300)

if __name__ == "__main__":
    visual_acc_from_file()