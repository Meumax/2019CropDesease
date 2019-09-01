# coding: utf-8
from torch.utils.data import *
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from augmentation import HorizontalFlip
    
class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):   
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])

normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def split_Dataset(data_dir, ratio, IMAGE_SIZE, train_transform=None, val_trainsform=None):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)     # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.imgs:  # 将数据按类标存放
        character[y].append(x)

    num_sample_train = int(len(character[0]) * ratio[0])

    train_inputs, val_inputs = [], []
    train_labels, val_labels = [], []
    
    for i, data in enumerate(character):   # data为一类图片
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:]:
            val_inputs.append(str(x))
            val_labels.append(i)
    
    train_dataset = MyDataset(train_inputs, train_labels, train_transform)
    val_dataset = MyDataset(val_inputs, val_labels, val_trainsform)
    
    return train_dataset, val_dataset



