import time
from torch.utils.data import DataLoader
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
from sganet.new import headCount_inceptionv3
import numpy as np
import torch.utils.data.sampler as sampler


SAVE_PATH = '/home/xuzhiwen/pythonProject/log2'

# save_path = os.path.join(SAVE_PATH, "ISANET-gk-nonpre-4.txt")
save_path = os.path.join(SAVE_PATH, "compare-gk-pre-isa-1.txt")


def my_print(str_log, save_path=save_path):
    if not isinstance(str_log, str):
        str_log = str(str_log)
    with open(save_path, "a+") as f:
        f.write(str_log + "\n")
    print(str_log)


# print(torch.hub.list('pytorch/vision'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)
print(device)

test_num = 443
val_num = 294
# test_num = 372
# val_num = 247
# RandomHorizontalFlip  按概率p=0.5水平翻转
# RandomVerticalFlip    按概率p=0.5垂直翻转
# Normalize             mean,std
transformation1 = transforms.Compose([transforms.Resize((299, 299)),
                                      # transforms.Grayscale(num_output_channels=1),
                                      transforms.ColorJitter(0.1),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

transformation2 = transforms.Compose([transforms.Resize((299, 299)),
                                      # transforms.Grayscale(num_output_channels=1),
                                      # transforms.ColorJitter(0.1),
                                      # transforms.RandomHorizontalFlip(),
                                      # transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataset_train = ImageFolder(r"/home/xuzhiwen/Data/train", transform=transformation1)
dataset_test = ImageFolder(r"/home/xuzhiwen/Data/train", transform=transformation2)
# dataset_train = ImageFolder(r"/home/xuzhiwen/chest-CT", transform=transformation1)
# dataset_test = ImageFolder(r"/home/xuzhiwen/chest-CT", transform=transformation2)

# trainingLoader = torch.utils.data.DataLoader(myDataset,batch_size=32,shuffle=True)
# myDataset[0]

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
                                           batch_size=8, shuffle=False, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=8, shuffle=False, sampler=valid_sampler)

model = headCount_inceptionv3()
model.to(device)
loss_function = nn.CrossEntropyLoss()
pata = list(model.parameters())  # 查看net内的参数
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.96)
best_acc = 0.0
best_epoch = 0
MAX_EPOCH = 200

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 0.75 ** (epoch//5)))
# /home/xuzhiwen/.cache/torch/hub/checkpoints
my_print("ISANET  299*299 batch_size = 8 , 0.4均分  1e-4")
for epoch in range(MAX_EPOCH):
    # train
    acc_train = 0.0
    model.train()  # 在训练过程中调用dropout方法
    running_loss = 0.0
    t1 = time.perf_counter()  # 统计训练一个epoch所需时间
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()

        outputs = model(images.to(device))
        #

        predict_y = torch.max(outputs, dim=1)[1]
        # predict_y = torch.argmax(outputs, dim=0)
        # print(labels.to(device))
        # print(labels.to(device).size())
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
    my_print(train_log)


    model.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in test_loader:
            test_images, test_labels = data_test
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            best_epoch = epoch
            # torch.save(model.state_dict(), save_path)

        print('[epoch %d] test_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))
        test_log = "test [%d/%d] loss: %.3f, acc %.4f, best_acc %.4f\n" % (epoch + 1, MAX_EPOCH, running_loss / step, acc / val_num, best_acc)
        my_print(test_log)

print('Finished Training')
