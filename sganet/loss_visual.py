# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/11/1 16:57
# @File     : acc_visual.py
# @Software : PyCharm

import os
import matplotlib.pyplot as plt
def visual_loss_from_file():
    # if not os.path.exists(file_path):
    #     assert "not find {0}!".format(file_path)
    file_path1 = ""
    # file_path2 = "models/4.21_Unet/CCM_fashion_weights/Adam0.001_batch64_L1/log.txt"
    # file_path3 = "models/4.21_Unet/HCM_fashion_weights/Adam0.001_batch64_L1/log.txt"
    with open(file_path1, "r", encoding='UTF-8') as f:
        datas1 = [list(map(str.strip, i.split(","))) for i in filter(lambda x: "train" in x, f.readlines())]
    # with open(file_path2, "r") as f:
    #     datas2 = [list(map(str.strip, i.split(","))) for i in filter(lambda x: "eval" in x, f.readlines())]
    # with open(file_path3, "r") as f:
    #     datas3 = [list(map(str.strip, i.split(","))) for i in filter(lambda x: "eval" in x, f.readlines())]

    loss = [float(i[0].split(" ")[-1]) for i in datas1]
    #print(loss)
    # MSED_cc = [float(i[1].split(" ")[1]) for i in datas2]
    # Unet_cc = [float(i[1].split(" ")[1]) for i in datas3]
    # print(len(total_loss))
    plt.plot([i for i in range(len(loss))],loss, linewidth = 0.5, color ='b', label = "Training")
    # plt.plot([i for i in range(len(MSED_cc))], MSED_cc, linewidth = 0.5, color='g', label="Pak et al. [8]")
    # plt.plot([i for i in range(len(MSED_cc))], Unet_cc, linewidth = 0.5, color='r', label="H. N. Abdullah et al. [4]")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.legend()

    #plt.show()
    plt.savefig("Ablation/inception1.jpg", dpi = 300)

if __name__ == "__main__":
    visual_loss_from_file()