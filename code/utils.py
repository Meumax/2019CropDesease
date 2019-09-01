# coding: utf-8
import os
import torch

'''
训练过程中保存loss和acc
'''
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return float(self.total_value)/ self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)

'''
保存训练快照
'''
def snapshot(savepathPre,fileName,state):    
    if not os.path.exists(savepathPre):
        os.makedirs(savepathPre)
    torch.save(state, savepathPre+fileName)

'''
预测data在model上的结果
输出两个GPU上的数组 所有的label值和所有的预测值
'''
def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            all_labels.append(labels)
            inputs = inputs.cuda()
            outputs = model(inputs)
            all_outputs.append(outputs.data.cpu())
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()
    return all_labels, all_outputs 

'''
将 a 扩展一维  并和acc在最后一维上做连接操作 
'''
def safe_stack_2array(acc, a):
    a = a.unsqueeze(-1) # 在最后一维扩充
    if acc is None:
        return a
    return torch.cat((acc, a), dim=acc.dim() - 1)

'''
TTA时使用不同的augmentation方法生成不同的dataLoader 并将预测结果连接
'''
def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        print('predict finish')
        prediction = safe_stack_2array(prediction, px)
    return lx, prediction











