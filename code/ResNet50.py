# coding: utf-8
# USAGE : python ResNet50.py --epoch 60 --modelPath ../model/

import torch
import CropModel
import CropDataset 
import torch.nn as nn
import torch.optim as optim
import utils
import os
import datetime
from torch.utils.data import *
import argparse
from tensorboardX import SummaryWriter

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("--epoch",type=int)
ap.add_argument("--modelPath",default="")
args = vars(ap.parse_args())

# 获取输入的参数
trainEpoch = args["epoch"]
modelPath = args["modelPath"]

# 定义模型参数
NB_CLASS=6
BATCH_SIZE=16
data_dir = "../dataset"
train_val_ratio = [0.7,0.3]
IMAGE_SIZE = 420
trian_transform = CropDataset.preprocess_with_augmentation(CropDataset.normalize_torch,IMAGE_SIZE)
val_transform = CropDataset.preprocess(CropDataset.normalize_torch,IMAGE_SIZE)
SEED=888
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True

# 获取当前日期
date=str(datetime.date.today())

def getmodel():
    print('[+] loading model ResNet50...', end='', flush=True)
    model=CropModel.resnet50_finetune(NB_CLASS)
    model.cuda()
    print('Done')
    return model

def train(epochNum):
    writer=SummaryWriter('../log/'+date+'/ResNet50/') # 创建 /log/日期/ResNet50的组织形式
    train_dataset,val_dataset = CropDataset.split_Dataset(data_dir, train_val_ratio, IMAGE_SIZE,trian_transform, val_transform)
    train_dataLoader = DataLoader(train_dataset,BATCH_SIZE,num_workers=16, shuffle=True)
    val_dataLoader = DataLoader(val_dataset,BATCH_SIZE,num_workers=1, shuffle=False)
    model = getmodel()
    criterion = nn.CrossEntropyLoss().cuda()
    min_loss=4.1
    print('min_loss is :%f'%(min_loss))
    min_acc=0.80
    patience=0
    lr=0.0
    momentum=0.0
    for epoch in range(epochNum):
        print('Epoch {}/{}'.format(epoch, epochNum - 1))
        print('-' * 10)
        
        #第一轮首先训练全连接层
        if epoch==0 or epoch==1 or epoch==2: 
            lr=1e-3
            optimizer = torch.optim.Adam(model.fresh_params(),lr = lr,amsgrad=True,weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(),lr = lr,amsgrad=True,weight_decay=1e-4)              
        if epoch==3:
            lr=1e-3
            momentum=0.9
            print('set lr=:%f,momentum=%f'%(lr,momentum))
        if patience==2 and lr==1e-3:
            patience=0
            model.load_state_dict(torch.load('../model/ResNet50/'+date+'_loss_best.pth')['state_dict'])
            lr=lr/10
            print('loss has increased lr divide 10 lr now is :%f'%(lr))
        if patience==2 and lr==1e-4:
            patience=0
            epochNum=epoch+1
        
        # 保存训练过程中的loss和acc
        running_loss = utils.RunningMean()
        running_corrects = utils.RunningMean()
            
        for batch_idx, (inputs, labels) in enumerate(train_dataLoader):    
            model.train(True) # 模型进入训练模式
            n_batchsize=inputs.size(0)
            optimizer.zero_grad()   # 清空所有参数的梯度
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss.update(loss.item(),1)  # 将这一个batch的loss保存起来
            _, preds = torch.max(outputs.data, 1)
            running_corrects.update(torch.sum(preds == labels.data).data,n_batchsize)   # 将这个batch的准确度保存起来
            loss.backward()
            optimizer.step()
            
            # 每10个batch显示一次训练结果信息
            if batch_idx%10==9:
                print('(%s)[epoch:%d,batch:%d]:acc: %f,loss:%f'%(str(datetime.datetime.now()),epoch,batch_idx,running_corrects.value,running_loss.value))
                niter = epoch * len(train_dataset)/BATCH_SIZE + batch_idx
                writer.add_scalar('Train/Acc',running_corrects.value,niter)
                writer.add_scalar('Train/Loss',running_loss.value,niter)
                # 如果batch大于300，则每300个batch进行一次验证               
                if batch_idx%300==299: 
                    lx,px=utils.predict(model,val_dataLoader)
                    log_loss = criterion(px,lx)
                    log_loss = log_loss.item()
                    _, preds = torch.max(px, dim=1)
                    accuracy = torch.mean((preds == lx).float())
                    writer.add_scalar('Val/Acc',accuracy,niter)
                    writer.add_scalar('Val/Loss',log_loss,niter)
                    print('(%s)[epoch:%d,batch:%d]: val_acc:%f,val_loss:%f,val_total_len:%d'%(epoch,batch_idx,accuracy,log_loss,len(val_dataset)))
        print('(%s)[epoch:%d] :acc: %f,loss:%f,lr:%f,patience:%d'%(str(datetime.datetime.now()),epoch,running_corrects.value,running_loss.value,lr,patience))       
        
        # 训练完后进行验证集验证
        lx,px=utils.predict(model,val_dataLoader)
        log_loss = criterion(px,lx)
        log_loss = log_loss.item()
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        writer.add_scalar('Val/Acc',accuracy,(epoch+1) * len(train_dataset)/BATCH_SIZE)
        writer.add_scalar('Val/Loss',log_loss,(epoch+1) * len(train_dataset)/BATCH_SIZE)
        print('(%s)[epoch:%d]: val_acc:%f,val_loss:%f,'%(str(datetime.datetime.now()),epoch,accuracy,log_loss))
        
        # 若验证集误差小于设定的min_loss,则保存模型快照
        if  log_loss < min_loss:
            try:
                fileName = date+'_loss_best.pth'
                utils.snapshot('../model/ResNet50/', fileName, {
                       'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'val_loss': log_loss,
                       'val_correct':accuracy })          
                patience = 0
                min_loss=log_loss
                print('save new model loss,now loss is ',min_loss)
            except IOError:
                print("Error: 没有找到文件或读取文件失败")    
        else:
            patience += 1    
            
        # 若精确度大于设定的min+acc,则保存模型快照    
        if accuracy>min_acc:
            try:
                fileName = date+'_acc_best.pth'
                utils.snapshot('../model/ResNet50/', fileName, {
                       'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'val_loss': log_loss,
                       'val_correct':accuracy })             
                min_acc=accuracy
                print('save new model acc,now acc is ',min_acc.item())    
            except IOError:
                print("Error: 没有找到文件或读取文件失败")

def trainWithRawData(path,epochNum):
    try:
        print('[+] loading modelParams...', end='', flush=True)
        modelParams=torch.load(path)
        print('Done')
    except IOError:
        print("Error: 没有找到文件或读取文件失败")
    writer=SummaryWriter('../log/'+date+'/ResNet50/') # 创建 /log/日期/ResNet50的组织形式
    train_dataset,val_dataset = CropDataset.split_Dataset(data_dir, train_val_ratio, IMAGE_SIZE,trian_transform, val_transform)
    train_dataLoader = DataLoader(train_dataset,BATCH_SIZE,num_workers=16, shuffle=True)
    val_dataLoader = DataLoader(val_dataset,BATCH_SIZE,num_workers=1, shuffle=False)
    model = getmodel()
    criterion = nn.CrossEntropyLoss().cuda()    
    model.load_state_dict(modelParams['state_dict'])
    min_loss=modelParams['val_loss']
    print('val_correct is %f'%(modelParams['val_correct'])) 
    print('min_loss is :%f'%(min_loss))
    min_acc=max(modelParams['val_correct'],0.81)
    optinizerSave=modelParams['optimizer']
    patience=0
    lr=1e-4
    momentum=0.9
    for epoch in range(epochNum):
        print('Epoch {}/{}'.format(epoch, epochNum - 1))
        print('-' * 10)
        if patience==3:
            patience=0
            model.load_state_dict(torch.load('../model/ResNet50/'+date+'_loss_best.pth')['state_dict'])
            lr=lr/5
            print('loss has increased , lr now is :%f'%(lr))
            optimizer=torch.optim.SGD(params=model.parameters(),lr=lr,momentum=0.9)       
        else:
            optimizer=torch.optim.SGD(params=model.parameters(),lr=lr,momentum=0.9)

        # 保存训练过程中的loss和acc
        running_loss = utils.RunningMean()
        running_corrects = utils.RunningMean()

        for batch_idx, (inputs, labels) in enumerate(train_dataLoader):
            model.train(True)
            n_batchsize=inputs.size(0)
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss.update(loss.item(),1)
            running_corrects.update(torch.sum(preds == labels.data).data,n_batchsize)
            loss.backward()
            optimizer.step()
            # 每10个batch显示一次训练结果信息
            if batch_idx%10==9:
                print('(%s)[epoch:%d,batch:%d]:acc: %f,loss:%f'%(str(datetime.datetime.now()),epoch,batch_idx,running_corrects.value,running_loss.value))
                niter = epoch * len(train_dataset)/BATCH_SIZE + batch_idx
                writer.add_scalar('Train/Acc',running_corrects.value,niter)
                writer.add_scalar('Train/Loss',running_loss.value,niter)
                # 如果batch大于300，则每300个batch进行一次验证               
                if batch_idx%300==299: 
                    lx,px=utils.predict(model,val_dataLoader)
                    log_loss = criterion(px,lx)
                    log_loss = log_loss.item()
                    _, preds = torch.max(px, dim=1)
                    accuracy = torch.mean((preds == lx).float())
                    writer.add_scalar('Val/Acc',accuracy,niter)
                    writer.add_scalar('Val/Loss',log_loss,niter)
                    print('(%s)[epoch:%d,batch:%d]: val_acc:%f,val_loss:%f,val_total_len:%d'%(epoch,batch_idx,accuracy,log_loss,len(val_dataset)))
        print('(%s)[epoch:%d] :acc: %f,loss:%f,lr:%f,patience:%d'%(str(datetime.datetime.now()),epoch,running_corrects.value,running_loss.value,lr,patience))       
        
        # 训练完后进行验证集验证
        lx,px=utils.predict(model,val_dataLoader)
        log_loss = criterion(px,lx)
        log_loss = log_loss.item()
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        writer.add_scalar('Val/Acc',accuracy,(epoch+1) * len(train_dataset)/BATCH_SIZE)
        writer.add_scalar('Val/Loss',log_loss,(epoch+1) * len(train_dataset)/BATCH_SIZE)
        print('(%s)[epoch:%d]: val_acc:%f,val_loss:%f,'%(str(datetime.datetime.now()),epoch,accuracy,log_loss))
        
        # 若验证集误差小于设定的min_loss,则保存模型快照
        if  log_loss < min_loss:
            try:
                fileName = date+'_loss_best.pth'
                utils.snapshot('../model/ResNet50/', fileName, {
                       'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'val_loss': log_loss,
                       'val_correct':accuracy })          
                patience = 0
                min_loss=log_loss
                print('save new model loss,now loss is ',min_loss)
            except IOError:
                print("Error: 没有找到文件或读取文件失败")    
        else:
            patience += 1    
            
        # 若精确度大于设定的min+acc,则保存模型快照    
        if accuracy>min_acc:
            try:
                fileName = date+'_acc_best.pth'
                utils.snapshot('../model/ResNet50/', fileName, {
                       'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'val_loss': log_loss,
                       'val_correct':accuracy })             
                min_acc=accuracy
                print('save new model acc,now acc is ',min_acc.item())    
            except IOError:
                print("Error: 没有找到文件或读取文件失败")

if __name__=="__main__":
    if modelPath=="":
        train(trainEpoch)
    else:
        trainWithRawData(modelPath,trainEpoch)