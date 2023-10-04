import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# 设置随机数种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# PCA 降维
def applyPCA(data, n_components):
    h, w, b = data.shape
    pca = PCA(n_components=n_components)
    data = np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))
    return data

# 创建 Dataset
class HXDataset(Dataset):
    def __init__(self, hsi, X, pos, windowSize, gt=None, transform=None, train=False):
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.X = None
        if(len(X.shape) == 2):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif(len(X.shape) == 3):
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])
        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi = self.transform(hsi).float()
            X = self.transform(X).float()
            trans = [transforms.RandomHorizontalFlip(1.),
                     transforms.RandomVerticalFlip(1.)]
            if self.train:
                if random.random() < 0.5:
                    i = random.randint(0, 1)
                    hsi = trans[i](hsi)
                    X = trans[i](X)
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            return hsi.unsqueeze(0), X, gt
        return hsi.unsqueeze(0), X, h, w

    def __len__(self):
        return self.pos.shape[0]

# 根据 index 获取数据
def getData(hsi_path, X_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers):
    '''
    hsi: 高光谱图像数据
    X: X 图像数据
    gt: 真实标签, 0 代表未标注
    train_index: 用于训练的数据索引
    test_index: 用于测试的数据索引
    trntst_index: 用于训练和测试的数据索引，用于对有标签的数据进行可视化
    all_index: 所有数据的索引，包含未标注数据，用于对所有数据进行可视化
    '''
    hsi = loadmat(hsi_path)[keys[0]]
    X = loadmat(X_path)[keys[1]]
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]
    # 使用 PCA 对 HSI 进行降维
    hsi = applyPCA(hsi, channels)
    # 创建 Dataset, 用于生成对应的 Dataloader
    HXtrainset = HXDataset(hsi, X, train_index,
                           windowSize, gt, transform=ToTensor(), train=True)
    HXtestset = HXDataset(hsi, X, test_index,
                          windowSize, gt, transform=ToTensor())
    HXtrntstset = HXDataset(hsi, X, trntst_index,
                            windowSize, transform=ToTensor())
    HXallset = HXDataset(hsi, X, all_index,
                         windowSize, transform=ToTensor())
    # 创建 Dataloader
    '''
    train_loader: 训练集
    test_loader: 测试集 
    trntst_loader: 用于画图，底色为白色，如 Trento 可视化图
    all_loader: 用于画图，底色为非白色，如 Houston 可视化图
    '''
    train_loader = DataLoader(
        HXtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(
        HXtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    trntst_loader = DataLoader(
        HXtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    all_loader = DataLoader(
        HXallset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    print("Success!")
    return train_loader, test_loader, trntst_loader, all_loader

# 获取 Houston2013 数据集
def getHouston2013Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Houston2013!")
    houston2013_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, houston2013_keys, channels, windowSize, batch_size, num_workers)

# 获取 Houston2018 数据集
def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Houston2018!")
    houston2018_keys = ['houston_hsi', 'houston_lidar', 'houston_gt', 'houston_train', 'houston_test', 'houston_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, houston2018_keys, channels, windowSize, batch_size, num_workers)

# 获取 Trento 数据集
def getTrentoData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Trento!")
    trento_keys = ['trento_hsi', 'trento_lidar', 'trento_gt', 'trento_train', 'trento_test', 'trento_all']
    return getData(hsi_path, lidar_path, gt_path, index_path, trento_keys, channels, windowSize, batch_size, num_workers)

# 获取 Berlin 数据集
def getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Berlin!")
    berlin_keys = ['berlin_hsi', 'berlin_sar', 'berlin_gt', 'berlin_train', 'berlin_test', 'berlin_all']
    return getData(hsi_path, sar_path, gt_path, index_path, berlin_keys, channels, windowSize, batch_size, num_workers)

# 获取 Augsburg 数据集
def getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    print("Augsburg!")
    augsburg_keys = ['augsburg_hsi', 'augsburg_sar', 'augsburg_gt', 'augsburg_train', 'augsburg_test', 'augsburg_all']
    return getData(hsi_path, sar_path, gt_path, index_path, augsburg_keys, channels, windowSize, batch_size, num_workers)

def getMyData(datasetType, channels, windowSize, batch_size, num_workers):
    if(datasetType == 0):
        hsi_path = '../data/Houston2013/houston_hsi.mat'
        lidar_path = '../data/Houston2013/houston_lidar.mat'
        gt_path = '../data/Houston2013/houston_gt.mat'
        index_path = '../data/Houston2013/houston_index.mat'
        train_loader, test_loader, trntst_loader, all_loader = getHouston2013Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 1):
        hsi_path = '../data/Houston2018/houston_hsi.mat'
        lidar_path = '../data/Houston2018/houston_lidar.mat'
        gt_path = '../data/Houston2018/houston_gt.mat'
        index_path = '../data/Houston2018/houston_index.mat'
        train_loader, test_loader, trntst_loader, all_loader = getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 2):    
        hsi_path = '../data/Trento/trento_hsi.mat'
        lidar_path = '../data/Trento/trento_lidar.mat'
        gt_path = '../data/Trento/trento_gt.mat'
        index_path = '../data/Trento/trento_index.mat'
        train_loader, test_loader, trntst_loader, all_loader = getTrentoData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 3):    
        hsi_path = "../data/Berlin/berlin_hsi.mat"
        sar_path = "../data/Berlin/berlin_sar.mat"
        gt_path = "../data/Berlin/berlin_gt.mat"
        index_path = "../data/Berlin/berlin_index.mat"
        train_loader, test_loader, trntst_loader, all_loader = getBerlinData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    elif(datasetType == 4):    
        hsi_path = "../data/Augsburg/augsburg_hsi.mat"
        sar_path = "../data/Augsburg/augsburg_sar.mat"
        gt_path = "../data/Augsburg/augsburg_gt.mat"
        index_path = "../data/Augsburg/augsburg_index.mat"
        train_loader, test_loader, trntst_loader, all_loader = getAugsburgData(hsi_path, sar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers)
    return train_loader, test_loader, trntst_loader, all_loader
