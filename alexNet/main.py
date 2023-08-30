import os
import time
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
import numpy as np
import torch.utils.data.sampler as sampler
from alexNet.alexnet import alexnet

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
# from sklearn.metrics import precision_recall_curve, average_precision_score


print(torch.__version__)
print(torch.cuda.is_available())
SAVE_PATH = 'E:\deeplearningwork\my-CNN-coding\\alexNet\metric'
# save_path = os.path.join(SAVE_PATH, "log-inception-ours-4.txt")
# save_path = os.path.join(SAVE_PATH, "inception-gk-nonpre-1.txt")
save_path = os.path.join(SAVE_PATH, "alexnet-our-2.txt")
save_path2 = os.path.join(SAVE_PATH, "alexnet-our-mertic-2.txt")

labellist = ['Adenocarcinoma', 'Normal', 'Squamous-Cell-Carcinoma']


def confusion_matrix(logits, labels, conf_matrix):
    # print(logits)
    # preds = torch.argmax(logits, 1)
    # labels = torch.argmax(labels, 1)
    preds = logits
    for p, t in zip(preds, labels):
        conf_matrix[p.long(), t.long()] += 1
    return conf_matrix


def my_print(str_log, save_path):
    if not isinstance(str_log, str):
        str_log = str(str_log)
    with open(save_path, "a+") as f:
        f.write(str_log + "\n")
    print(str_log)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_num = 443
val_num = 294
# test_num = 372
# val_num = 247
# RandomHorizontalFlip  按概率p=0.5水平翻转
# RandomVerticalFlip    按概率p=0.5垂直翻转
# Normalize             mean,std
transformation1 = transforms.Compose([transforms.Resize((256, 256)),
                                      # transforms.Grayscale(num_output_channels=1),
                                      transforms.CenterCrop(224),
                                      # transforms.CenterCrop(112),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transformation2 = transforms.Compose([transforms.Resize((256, 256)),
                                      # transforms.CenterCrop(112),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_train = ImageFolder("E:\deeplearningwork\my-CNN-coding\\archive", transform=transformation1)
dataset_test = ImageFolder("E:\deeplearningwork\my-CNN-coding\\archive", transform=transformation2)
# dataset_train = ImageFolder(r"/home/xuzhiwen/chest-CT", transform=transformation1)
# dataset_test = ImageFolder(r"/home/xuzhiwen/chest-CT", transform=transformation2)


test_size = 0.4
samples = len(dataset_train)
indices = list(range(samples))
np.random.shuffle(indices)
train_len = int(np.floor(samples * (test_size)))
train_idx, valid_idx = indices[train_len:], indices[:train_len]
train_sampler = sampler.SubsetRandomSampler(train_idx)
valid_sampler = sampler.SubsetRandomSampler(valid_idx)
print(len(train_sampler), len(valid_sampler))

train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=2, shuffle=False, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=2, shuffle=False, sampler=valid_sampler)

# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model = alexnet()
# fc_features = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(fc_features, 3)
model.to(device)

loss_function = nn.CrossEntropyLoss()
pata = list(model.parameters())  # 查看net内的参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.96)
best_acc = 0.0
best_epoch = 0
MAX_EPOCH = 50
pre = [0.0, 0.0, 0.0]
rec = [0.0, 0.0, 0.0]
f1 = [0.0, 0.0, 0.0]
my_print("alex 256*256 batch_size = 8 , 0.4均分  1e-5", save_path)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 0.8 ** (epoch//5)))
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
for epoch in range(MAX_EPOCH):
    # train
    acc_train = 0.0
    model.train()  # 在训练过程中调用dropout方法.
    running_loss = 0.0
    valid_loss = 0.0
    t1 = time.perf_counter()  # 统计训练一个epoch所需时间
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()

        outputs = model(images.to(device))
        # outputs = outputs.logits
        predict_y = torch.max(outputs, dim=1)[1]
        pre_x = predict_y
        acc_train += (predict_y == labels.to(device)).sum().item()
        #
        loss = loss_function(outputs, labels.to(device))

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

    # print(time.perf_counter()-t1)
    # scheduler.step()
    # print("当前学习率：", scheduler.get_last_lr())
    print()
    print("epoch:{}, acc_train:{}".format(epoch, acc_train / test_num))

    train_log = "train [%d/%d] loss %.4f, acc %.4f, best_epoch %d\n" % (
        epoch, MAX_EPOCH, running_loss / test_num, acc_train / test_num, best_epoch)
    my_print(train_log, save_path)

    conf_matrix = torch.zeros(3, 3)
    model.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
    acc = 0.0  # accumulate accurate number / epoch
    predict_yy = 0
    true_yy = 0

    with torch.no_grad():
        for data_test in test_loader:
            test_images, test_labels = data_test
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            # print(predict_y)
            predict_yy = predict_y
            true_yy = test_labels
            acc += (predict_y == test_labels.to(device)).sum().item()
            # conf_matrix = confusion_matrix(predict_y, test_labels, conf_matrix)
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            best_epoch = epoch
            # torch.save(model.state_dict(), save_path)
        print('[epoch %d] test_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))

        score_list = []  # 存储预测得分
        label_list = []  # 存储真实标签
        num_class = 3
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
            score_tmp = outputs  # (batchsize, nclass)

            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())

        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
        # print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
        # print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum]) onehot

        precision_dict = dict()
        recall_dict = dict()
        average_precision_dict = dict()

        for i in range(num_class):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
            average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
            print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

        # micro
        precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                                  score_array.ravel())
        average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
        print('AlexNet: {0:0.2f}'.format(
            average_precision_dict["micro"]))
        print("recall   ", recall_dict['micro'])
        print()
        print("precision  ", precision_dict['micro'])
        # 绘制所有类别平均的pr曲线
        plt.figure()
        plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'AlexNet: AP={0:0.2f}'
            .format(average_precision_dict["micro"]))
        # plt.show()
        plt.savefig("/home/xuzhiwen/pythonProject/right/metric/alex/gk-alex-pr-" + str(epoch + 1) + ".png")

        # df_cm = pd.DataFrame(conf_matrix.numpy(),
        #                      index=[i for i in list(labellist)],
        #                      columns=[i for i in list(labellist)]
        #                      )
        # plt.figure(figsize=(10, 7))
        # sn.heatmap(df_cm, annot=True, fmt=".1f", cmap="BuPu")
        # # plt.show()
        # plt.savefig("/home/xuzhiwen/pythonProject/right/alex/gk-confusion"+str(epoch+1)+".png", dpi=300)
        #
        # n1 = len(conf_matrix)
        # precision = []
        # recall = []
        # f1score = []
        # metric_log = ''
        #
        # for i in range(n1):
        #     rowsum, colsum = sum(conf_matrix[i]), sum(conf_matrix[r][i] for r in range(n1))
        #     try:
        #         precision.append(conf_matrix[i][i] / float(colsum))
        #         recall.append(conf_matrix[i][i] / float(rowsum))
        #         f1score.append(
        #             2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0)
        #         print(
        #             labellist[i] + ":  precision: {:.4f}  recall: {:.4f}  f1: {:.4f} ".format(precision[i], recall[i], f1score[i]))
        #         metric_log += "["+str(epoch + 1) +"]  " +labellist[i] + ":  precision: {:.4f}  recall: {:.4f}  f1: {:.4f} \n".format(precision[i], recall[i], f1score[i])
        #         pre[i] += precision[i]
        #         rec[i] += recall[i]
        #         f1[i] += f1score[i]
        #
        #         avg_pre = np.array(pre) / (epoch + 1)
        #         avg_rec = np.array(rec) / (epoch + 1)
        #         avg_f1 = np.array(f1) / (epoch + 1)
        #         print(labellist[i] + ": avg_precision: {:.4f}  avg_recall: {:.4f}  avg_f1: {:.4f} ".format(avg_pre[i], avg_rec[i], avg_f1[i]))
        #     except ZeroDivisionError:
        #         precision.append(0)
        #         recall.append(0)
        #         f1score.append(0)
        #         print(
        #             "precision: {}  recall: {}  f1: {} ".format(0, 0, 0))
        #         metric_log += "["+str(epoch + 1) +"]  " + labellist[i] + "precision: {}  recall: {}  f1: {} \n".format(0, 0, 0)
        # for i in range(n1):
        #     metric_log += labellist[i] + ": avg_precision: {:.4f}  avg_recall: {:.4f}  avg_f1: {:.4f} \n".format(avg_pre[i],
        #                                                                                                    avg_rec[i],
        #                                                                                                    avg_f1[i])
        # correct1 = [conf_matrix[i][i] for i in range(len(labellist))]
        # total_acc1 = sum(correct1) / sum(map(sum, conf_matrix))
        # print("total accuracy: {:.4} ".format(total_acc1))
        # metric_log += "total accuracy: {:.4} \n".format(total_acc1)

        test_log = "test [%d/%d] loss: %.3f, acc %.4f, best_acc %.4f\n" % (
            epoch + 1, MAX_EPOCH, running_loss / step, acc / val_num, best_acc)
        my_print(test_log, save_path)
        # my_print(metric_log, save_path=save_path2)
print('Finished Training')
