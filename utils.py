import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import ast
import torch.nn as nn
import torch.optim as optim
import random
from scipy.stats import pearsonr

class MyDataset(Dataset):
    def __init__(self,x_discrete,x_continuous,y1,y2):
        self.x_d=x_discrete
        self.x_c=x_continuous
        self.y1=y1
        self.y2=y2
    def __len__(self):
        return self.x_d.shape[0]
    def __getitem__(self,idx):
        return (self.x_d[idx],self.x_c[idx],self.y1[idx],self.y2[idx])

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 GPU，也需要固定 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
set_all_seed(42)

def load_exp1_dataset():
    # 前25个特征是离散特征，后13个特征是连续特征
    df_train = pd.read_csv("data/exp1/raw_train.csv")
    df_test = pd.read_csv("data/exp1/raw_test.csv")

    df_train['vectorized_features_1'] = df_train['vectorized_features_1'].apply(ast.literal_eval)
    df_test['vectorized_features_1'] = df_test['vectorized_features_1'].apply(ast.literal_eval)

    # 获取训练数据
    x_train = torch.from_numpy(np.array(df_train['vectorized_features_1'].values.tolist(), np.float32))
    x_train_d = x_train[:, :25].long()  # 前25维是离散变量(discrete)，要转为long类型才能送入embedding层
    x_train_c = x_train[:, 25:]  # 中间13维是连续变量(continuous), 最后一维是0或1，也当做连续变量
    y1_train = torch.from_numpy(np.array(df_train['income'].values.tolist(), np.float32))
    y2_train = torch.from_numpy(np.array(df_train['AMARITL'].values.tolist(), np.float32))

    # 计算两个任务的标签间的pearson相关系数
    corr, p_value = pearsonr(y1_train, y2_train)
    print(f"Absolute Pearson correlation coefficient: {abs(corr)}")  # 符合原论文的0.1768
    print(f"P-value: {p_value}")

    # 获取测试数据
    x_test = torch.from_numpy(np.array(df_test['vectorized_features_1'].values.tolist(), np.float32))
    x_test_d = x_test[:, :25].long()  # 前25维是离散变量(discrete)，要转为long类型才能送入embedding层
    x_test_c = x_test[:, 25:]  # 中间13维是连续变量(continuous), 最后一维是0或1，也当做连续变量
    y1_test = torch.from_numpy(np.array(df_test['income'].values.tolist(), np.float32))
    y2_test = torch.from_numpy(np.array(df_test['AMARITL'].values.tolist(), np.float32))

    # 训练集
    train_dataset = MyDataset(x_train_d, x_train_c, y1_train, y2_train)
    # 验证集和测试集按照原论文1:1
    val_dataset = MyDataset(x_test_d[:23695], x_test_c[:23695], y1_test[:23695], y2_test[:23695])
    test_dataset = MyDataset(x_test_d[23695:47390], x_test_c[23695:47390], y1_test[23695:47390], y2_test[23695:47390])

    return train_dataset, val_dataset, test_dataset

def load_exp2_dataset():
    # 前25个特征是离散特征，后13个特征是连续特征
    df_train = pd.read_csv("data/exp2/raw_train.csv")
    df_test = pd.read_csv("data/exp2/raw_test.csv")

    df_train['vectorized_features_2'] = df_train['vectorized_features_2'].apply(ast.literal_eval)
    df_test['vectorized_features_2'] = df_test['vectorized_features_2'].apply(ast.literal_eval)

    # 获取训练数据
    x_train = torch.from_numpy(np.array(df_train['vectorized_features_2'].values.tolist(), np.float32))
    x_train_d = x_train[:, :25].long()  # 前25维是离散变量(discrete)，要转为long类型才能送入embedding层
    x_train_c = x_train[:, 25:]  # 中间13维是连续变量(continuous), 最后一维是0或1，也当做连续变量
    y1_train = torch.from_numpy(np.array(df_train['AHSCOL'].values.tolist(), np.float32))
    y2_train = torch.from_numpy(np.array(df_train['AMARITL'].values.tolist(), np.float32))

    from scipy.stats import pearsonr
    corr, p_value = pearsonr(y1_train, y2_train)
    # 和原论文的0.2373不相同，主要是因为数据预处理时的细节不同
    # 我的做法是删除掉AHGA，然后把AHSCOL转化为0或1；原论文的做法可能是删除掉AHSCOL，把AHGA转化为0或1
    print(f"Absolute Pearson correlation coefficient: {abs(corr)}")
    print(f"P-value: {p_value}")

    # 获取测试数据
    x_test = torch.from_numpy(np.array(df_test['vectorized_features_2'].values.tolist(), np.float32))
    x_test_d = x_test[:, :25].long()  # 前25维是离散变量(discrete)，要转为long类型才能送入embedding层
    x_test_c = x_test[:, 25:]  # 中间13维是连续变量(continuous), 最后一维是0或1，也当做连续变量
    y1_test = torch.from_numpy(np.array(df_test['AHSCOL'].values.tolist(), np.float32))
    y2_test = torch.from_numpy(np.array(df_test['AMARITL'].values.tolist(), np.float32))

    # 训练集
    train_dataset = MyDataset(x_train_d, x_train_c, y1_train, y2_train)
    # 验证集和测试集按照原论文1:1
    val_dataset = MyDataset(x_test_d[:23695], x_test_c[:23695], y1_test[:23695], y2_test[:23695])
    test_dataset = MyDataset(x_test_d[23695:47390], x_test_c[23695:47390], y1_test[23695:47390], y2_test[23695:47390])

    return train_dataset, val_dataset, test_dataset