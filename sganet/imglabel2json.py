import json
import os
import numpy as np
import cv2

def get_json(phase, fold):
    img_path='/home/zhangxiangbo/ML-GCN/data/dataset_96_random_4/'+phase+'_data_50fold'+str(fold)+'/'
    json_list = []
    for file in os.listdir('./dataset_96_random_4/'+phase+'_data_50fold'+str(fold)+'/'):

        dict_file = {}
        cat_np=np.zeros((5,))
        cat_list = []
        img = cv2.imread(img_path+file, cv2.IMREAD_GRAYSCALE)

        name = file
        malignancy=file.split('_')[-2]
        texture=file.split('_')[-3]
        lobulation=file.split('_')[-5]

        margin=file.split('_')[-6]
        sphericity=file.split('_')[-7]
        calcification=file.split('_')[-8]
        subtlety=file.split('_')[-10]

        cat_np[4]=round(float(texture)/5,2)
        cat_np[3]=round(float(margin)/5,2)
        cat_np[2]=round(float(sphericity)/5,2)
        cat_np[1]=round(float(calcification)/6,2)
        cat_np[0]=round(float(subtlety)/5,2)

        # cat_np[4]=float(texture)
        # cat_np[3]=float(margin)
        # cat_np[2]=float(sphericity)
        # cat_np[1]=float(calcification)
        # cat_np[0]=float(subtlety)

        cat_list.append(cat_np[0])
        cat_list.append(cat_np[1])
        cat_list.append(cat_np[2])
        cat_list.append(cat_np[3])
        cat_list.append(cat_np[4])

        # cat_np[cat_np>3]=1
        # cat_np[cat_np!=1]=0

        malignancy = int(malignancy)
        if(malignancy>3):
            malignancy=1
        elif(malignancy<3):
            malignancy=0
        else:
            continue
        # cat_list=cat_np.tolist()

        dict_file["file_name"] = name
        dict_file["label"] = cat_list
        dict_file["malignancy"] = malignancy
        json_list.append(dict_file)

    with open('./dataset_96_random_4/MSE_normal_image_'+phase+'_labelfold'+str(fold)+'.json','w') as file_obj:
        json.dump(json_list,file_obj)

for i in range(5):
    phase_train = 'train'
    phase_test = 'test'
    fold = i+1
    get_json(phase_train, fold)
    get_json(phase_test, fold)
