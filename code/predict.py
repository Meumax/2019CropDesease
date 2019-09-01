# coding: utf-8
# USAGE : python predict.py --mode all --input ../testset/danyi100 --output ../output/danyi_all.json

import torch
import utils
import CropModel
import random
import numpy as np
import os
import torch.nn.functional as F
import json
import argparse
import datetime
import sys
import collections

from torch.utils.data import DataLoader
from torchvision import transforms
from augmentation import five_crops, HorizontalFlip, make_transforms
from CropDataset import MyDataset, preprocess, preprocess_hflip, normalize_05, normalize_torch
from scipy.stats.mstats import gmean

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("--mode")
ap.add_argument("--input")
ap.add_argument("--output")
args = vars(ap.parse_args())

# 获取输入的参数
train_mode = args["mode"]
data_dir = args["input"]
json_file = args["output"]

NB_CLASS=6
BATCH_SIZE=32
SEED=888
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True

# 读取测试集
test_inputs = []
for root, dirs, files in os.walk(data_dir):
    path = [os.path.join(root, name) for name in files]
    test_inputs.extend(path)

test_labels = [0 for x in range(len(test_inputs))]

def get_model(model_class):
    print('[+] loading model... ', end='', flush=True)
    model = model_class(NB_CLASS)
    model.cuda()
    print('done')
    return model

# 直接使用原始数据进行预测
def predictRaw(model_name, model_class, weight_pth, image_size, normalize):
    print("[+] {0} predictRaw".format(model_name))
    model = get_model(model_class)
    model.load_state_dict(torch.load(weight_pth)['state_dict'])
    model.eval()
    print('{0} load state dict done'.format(model_name))
    
    test_dataset = MyDataset(test_inputs,test_labels,transform=preprocess(normalize, image_size))
    data_loader = DataLoader(dataset=test_dataset, num_workers=16,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    lx,px=utils.predict(model,data_loader)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    if not os.path.exists('../feature/'+model_name):
        os.makedirs('../feature/'+model_name)
    torch.save(data, '../feature/'+model_name+'/raw_prediction.pth')


# 原始数据进行数据增强后再预测(裁剪)
def predictCrop(model_name, model_class, weight_pth, image_size, normalize):
    print("[+] {0} predictCrop".format(model_name))
    model = get_model(model_class)
    model.load_state_dict(torch.load(weight_pth)['state_dict'])
    model.eval()
    print('{0} load state dict done'.format(model_name))
  
    tta_preprocess=[preprocess(normalize, image_size)]
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [transforms.ToTensor(), normalize],
                                      five_crops(image_size))

    print('[+] tta size: {0}'.format(len(tta_preprocess)))
    
    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = MyDataset(test_inputs,test_labels,transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=16,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)
        print('add transforms')

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    if not os.path.exists('../feature/'+model_name):
        os.makedirs('../feature/'+model_name)
    torch.save(data, '../feature/'+model_name+'/crop_prediction.pth')
    print('Done')

# 原始数据进行数据增强后再预测(翻转)
def predictFlip(model_name, model_class, weight_pth, image_size, normalize):
    print("[+] {0} predictFlip".format(model_name))
    model = get_model(model_class)
    model.load_state_dict(torch.load(weight_pth)['state_dict'])
    model.eval()
    print('{0} load state dict done'.format(model_name))

    tta_preprocess = [preprocess(normalize, image_size), preprocess_hflip(normalize, image_size)]

    print('[+] tta size: {0}'.format(len(tta_preprocess)))

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = MyDataset(test_inputs,test_labels,transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=16,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)
        print('add transforms')

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    if not os.path.exists('../feature/'+model_name):
        os.makedirs('../feature/'+model_name)
    torch.save(data, '../feature/'+model_name+'/flip_prediction.pth')
    print('Done')

# 原始数据进行数据增强后再预测(裁剪+翻转)
def predictAll(model_name, model_class, weight_pth, image_size, normalize):
    print("[+] {0} predictAll.".format(model_name))
    model = get_model(model_class)
    model.load_state_dict(torch.load(weight_pth)['state_dict'])
    model.eval()
    print('{0} load state dict done'.format(model_name))

    tta_preprocess = [preprocess(normalize, image_size), preprocess_hflip(normalize, image_size)]
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [transforms.ToTensor(), normalize],
                                      five_crops(image_size))
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [HorizontalFlip(), transforms.ToTensor(), normalize],
                                      five_crops(image_size))
    print('[+] tta size: {0}'.format(len(tta_preprocess)))

    data_loaders = []
    for transform in tta_preprocess:
        # test_dataset,_ = split_Dataset(data_dir, ratio, image_size, train_transform=transform)
        test_dataset = MyDataset(test_inputs,test_labels,transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=16,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)
        print('add transforms')

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    if not os.path.exists('../feature/'+model_name):
        os.makedirs('../feature/'+model_name)
    torch.save(data, '../feature/'+model_name+'/all_prediction.pth')
    print('{0} Predict Done'.format(model_name))

# 每个模型所有的预测手段全部使用
def All_Predict():
    predictAll('DenseNet121',CropModel.densenet121_finetune,'../model/DenseNet121/2019-08-09_acc_best.pth',224,normalize_torch)
    predictAll('DenseNet161',CropModel.densenet161_finetune,'../model/DenseNet161/2019-08-10_acc_best.pth',224,normalize_torch)
    predictAll('DenseNet201',CropModel.densenet201_finetune,'../model/DenseNet201/2019-08-10_acc_best.pth',224,normalize_torch)
    predictAll('Inception_v3',CropModel.InceptionV3Finetune,'../model/InceptionV3/2019-08-11_acc_best.pth',299,normalize_05)
    predictAll('Inception_v4',CropModel.inceptionv4_finetune,'../model/InceptionV4/2019-08-12_acc_best.pth',299,normalize_05)
    predictAll('InceptionresNet_v2',CropModel.inceptionresnetv2_finetune,'../model/InceptionresNet_v2/2019-08-12_acc_best.pth',299,normalize_05)
    predictAll('ResNet50',CropModel.resnet50_finetune,'../model/ResNet50/2019-08-14_acc_best.pth',420,normalize_torch)
    predictAll('ResNet101',CropModel.resnet101_finetune,'../model/ResNet101/2019-08-12_acc_best.pth',224,normalize_torch)
    predictAll('ResNet152',CropModel.resnet152_finetune,'../model/ResNet152/2019-08-13_acc_best.pth',224,normalize_torch)

def Raw_Predict():
    predictRaw('DenseNet121',CropModel.densenet121_finetune,'../model/DenseNet121/2019-08-09_acc_best.pth',224,normalize_torch)
    predictRaw('DenseNet161',CropModel.densenet161_finetune,'../model/DenseNet161/2019-08-10_acc_best.pth',224,normalize_torch)
    predictRaw('DenseNet201',CropModel.densenet201_finetune,'../model/DenseNet201/2019-08-10_acc_best.pth',224,normalize_torch)
    predictRaw('Inception_v3',CropModel.InceptionV3Finetune,'../model/InceptionV3/2019-08-11_acc_best.pth',299,normalize_05)
    predictRaw('Inception_v4',CropModel.inceptionv4_finetune,'../model/InceptionV4/2019-08-12_acc_best.pth',299,normalize_05)
    predictRaw('InceptionresNet_v2',CropModel.inceptionresnetv2_finetune,'../model/InceptionresNet_v2/2019-08-12_acc_best.pth',299,normalize_05)
    predictRaw('ResNet50',CropModel.resnet50_finetune,'../model/ResNet50/2019-08-14_acc_best.pth',420,normalize_torch)
    predictRaw('ResNet101',CropModel.resnet101_finetune,'../model/ResNet101/2019-08-12_acc_best.pth',224,normalize_torch)
    predictRaw('ResNet152',CropModel.resnet152_finetune,'../model/ResNet152/2019-08-13_acc_best.pth',224,normalize_torch)

def Crop_Predict():    
    predictCrop('DenseNet121',CropModel.densenet121_finetune,'../model/DenseNet121/2019-08-09_acc_best.pth',224,normalize_torch)
    predictCrop('DenseNet161',CropModel.densenet161_finetune,'../model/DenseNet161/2019-08-10_acc_best.pth',224,normalize_torch)
    predictCrop('DenseNet201',CropModel.densenet201_finetune,'../model/DenseNet201/2019-08-10_acc_best.pth',224,normalize_torch)
    predictCrop('Inception_v3',CropModel.InceptionV3Finetune,'../model/InceptionV3/2019-08-11_acc_best.pth',299,normalize_05)
    predictCrop('Inception_v4',CropModel.inceptionv4_finetune,'../model/InceptionV4/2019-08-12_acc_best.pth',299,normalize_05)
    predictCrop('InceptionresNet_v2',CropModel.inceptionresnetv2_finetune,'../model/InceptionresNet_v2/2019-08-12_acc_best.pth',299,normalize_05)
    predictCrop('ResNet50',CropModel.resnet50_finetune,'../model/ResNet50/2019-08-14_acc_best.pth',420,normalize_torch)
    predictCrop('ResNet101',CropModel.resnet101_finetune,'../model/ResNet101/2019-08-12_acc_best.pth',224,normalize_torch)
    predictCrop('ResNet152',CropModel.resnet152_finetune,'../model/ResNet152/2019-08-13_acc_best.pth',224,normalize_torch)

def Flip_Predict():       
    predictFlip('DenseNet121',CropModel.densenet121_finetune,'../model/DenseNet121/2019-08-09_acc_best.pth',224,normalize_torch)
    predictFlip('DenseNet161',CropModel.densenet161_finetune,'../model/DenseNet161/2019-08-10_acc_best.pth',224,normalize_torch)
    predictFlip('DenseNet201',CropModel.densenet201_finetune,'../model/DenseNet201/2019-08-10_acc_best.pth',224,normalize_torch)
    predictFlip('Inception_v3',CropModel.InceptionV3Finetune,'../model/InceptionV3/2019-08-11_acc_best.pth',299,normalize_05)
    predictFlip('Inception_v4',CropModel.inceptionv4_finetune,'../model/InceptionV4/2019-08-12_acc_best.pth',299,normalize_05)
    predictFlip('InceptionresNet_v2',CropModel.inceptionresnetv2_finetune,'../model/InceptionresNet_v2/2019-08-12_acc_best.pth',299,normalize_05)
    predictFlip('ResNet50',CropModel.resnet50_finetune,'../model/ResNet50/2019-08-14_acc_best.pth',420,normalize_torch)
    predictFlip('ResNet101',CropModel.resnet101_finetune,'../model/ResNet101/2019-08-12_acc_best.pth',224,normalize_torch)
    predictFlip('ResNet152',CropModel.resnet152_finetune,'../model/ResNet152/2019-08-13_acc_best.pth',224,normalize_torch)

def Load_Features(pth_file):
    DenseNet121_Pre = torch.load('../feature/DenseNet121/'+pth_file)['px']
    DenseNet161_Pre= torch.load('../feature/DenseNet161/'+pth_file)['px']
    DenseNet201_Pre = torch.load('../feature/DenseNet201/'+pth_file)['px']
    InceptionResNet_v2_Pre = torch.load('../feature/InceptionresNet_v2/'+pth_file)['px']
    Inception_v3_Pre = torch.load('../feature/Inception_v3/'+pth_file)['px']
    Inception_v4_Pre = torch.load('../feature/Inception_v4/'+pth_file)['px']
    ResNet50_Pre = torch.load('../feature/ResNet50/'+pth_file)['px']
    ResNet101_Pre = torch.load('../feature/ResNet101/'+pth_file)['px']
    ResNet152_Pre = torch.load('../feature/ResNet152/'+pth_file)['px']
    return DenseNet121_Pre,DenseNet161_Pre,DenseNet201_Pre,InceptionResNet_v2_Pre,Inception_v3_Pre,Inception_v4_Pre,ResNet50_Pre,ResNet101_Pre,ResNet152_Pre

if __name__=="__main__":
    start_time = datetime.datetime.now()
    
    print('')
    
    # 训练模式选择：all方式有12种增强手段，crop有6种，flip有2种，raw为不增强直接进行预测，增强手段越多耗时越长，准确度越高
    if train_mode == "all":
        print("Predict's mode is all_mode")
        All_Predict()
        DenseNet121_Pre, DenseNet161_Pre, DenseNet201_Pre, InceptionResNet_v2_Pre, Inception_v3_Pre, Inception_v4_Pre, ResNet50_Pre, ResNet101_Pre, ResNet152_Pre = Load_Features('all_prediction.pth')
    elif train_mode == "crop":
        print("Predict's mode is crop_mode")
        Crop_Predict()
        DenseNet121_Pre, DenseNet161_Pre, DenseNet201_Pre, InceptionResNet_v2_Pre, Inception_v3_Pre, Inception_v4_Pre, ResNet50_Pre, ResNet101_Pre, ResNet152_Pre = Load_Features('crop_prediction.pth')
    elif train_mode == "flip":
        print("Predict's mode is flip_mode")
        Flip_Predict()
        DenseNet121_Pre, DenseNet161_Pre, DenseNet201_Pre, InceptionResNet_v2_Pre, Inception_v3_Pre, Inception_v4_Pre, ResNet50_Pre, ResNet101_Pre, ResNet152_Pre = Load_Features('flip_prediction.pth')
    elif train_mode == "raw":
        print("Predict's mode is raw_mode")
        Raw_Predict()
        DenseNet121_Pre, DenseNet161_Pre, DenseNet201_Pre, InceptionResNet_v2_Pre, Inception_v3_Pre, Inception_v4_Pre, ResNet50_Pre, ResNet101_Pre, ResNet152_Pre = Load_Features('raw_prediction.pth')
        DenseNet121_Pre = torch.unsqueeze(DenseNet121_Pre,2)
        DenseNet161_Pre = torch.unsqueeze(DenseNet161_Pre,2)
        DenseNet201_Pre = torch.unsqueeze(DenseNet201_Pre,2)
        InceptionResNet_v2_Pre = torch.unsqueeze(InceptionResNet_v2_Pre,2)
        Inception_v3_Pre = torch.unsqueeze(Inception_v3_Pre,2)
        Inception_v4_Pre = torch.unsqueeze(Inception_v4_Pre,2)
        ResNet50_Pre = torch.unsqueeze(ResNet50_Pre,2)
        ResNet101_Pre = torch.unsqueeze(ResNet101_Pre,2)
        ResNet152_Pre = torch.unsqueeze(ResNet152_Pre,2)
        

    #将预测结果进行融合计算
    test_prob = F.softmax(torch.cat((DenseNet121_Pre,DenseNet161_Pre,DenseNet201_Pre,InceptionResNet_v2_Pre,Inception_v3_Pre,Inception_v4_Pre,ResNet50_Pre,ResNet101_Pre,ResNet152_Pre),dim=2),dim=1).numpy()
    test_prob = gmean(test_prob,axis=2)
    test_predict = np.argmax(test_prob,axis=1)

    print("预测输出的个数为：",len(test_predict))
    test_img_list = os.listdir(data_dir)

    result = []

    # 生成json文件
    for index,img_name in enumerate(test_img_list):
        test_predict[index] += 1
        a = collections.OrderedDict()
        a['image_id'] = img_name
        a['disease_class'] = int(test_predict[index])
        # a={'image_id':img_name,'disease_class':int(test_predict[index])}
        result.append(a.copy())
        
    with open(json_file,'w') as f:
        json.dump(result,f,ensure_ascii=False)
    
    end_time = datetime.datetime.now()
    time_consuming = (end_time - start_time).seconds
    print("预测结束，耗时{0}秒".format(time_consuming))
