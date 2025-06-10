import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from mozi_ai_sdk.FKFD.dataProcess import *
from PySide2.QtCore import QObject, Signal, Slot

mpl.use('TkAgg')  # !IMPORTANT
mpl.rcParams['font.sans-serif'] = ['SimHei']

device = torch.device("cpu")


class wavelet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        激活函数：wavelet小波基函数
        :param ctx: 11
        :param input: 输入
        :return:
        """
        ctx.input = input
        return (torch.cos(1.75 * input)) * (torch.exp((input ** 2) / (-2)))

    @staticmethod
    def backward(ctx, grad_output):
        """
        梯度反传
        :param ctx:
        :param grad_output:
        :return:
        """
        input = ctx.input

        return grad_output * (-1) * (1.75 * torch.sin(1.75 * input) * (torch.exp((input ** 2) / (-2)))
                                     + grad_output * torch.cos(1.75 * input)) * (-1) * input * (
                   torch.exp((input ** 2) / (-2)))


class WNNDDDDNet(nn.Module):
    """
    适用于DDDD数据的小波神经网络威胁评估模型
    """

    def __init__(self):
        super(WNNDDDDNet, self).__init__()

        # input layer
        self.fc1 = nn.Linear(7, 84)

        # 定义可训练参数a和b
        self.a = nn.Parameter(torch.Tensor(7))
        self.b = nn.Parameter(torch.Tensor(7))
        # 正态分布初始化参数a和b
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)
        self.relu = nn.ReLU()
        # output layer
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = (x - self.b) / self.a
        x = wavelet.apply(x)
        # pass through input layer
        x = self.fc1(x)

        x = self.relu(x)
        # pass through output layer
        x = self.fc3(x)

        return x


class WNNMissileNet(nn.Module):
    """
    适用于其他DD数据的小波神经网络威胁评估模型
    """

    def __init__(self):
        super(WNNMissileNet, self).__init__()

        # input layer
        self.fc1 = nn.Linear(8, 96)

        # 定义可训练参数a和b
        self.a = nn.Parameter(torch.Tensor(8))
        self.b = nn.Parameter(torch.Tensor(8))
        # 正态分布初始化参数a和b
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)
        self.relu = nn.ReLU()
        # output layer
        self.fc3 = nn.Linear(96, 1)

    def forward(self, x):
        x = (x - self.b) / self.a
        x = wavelet.apply(x)
        # pass through input layer
        x = self.fc1(x)

        x = self.relu(x)
        # pass through output layer
        x = self.fc3(x)

        return x


class WNNAttackPlaneNet(nn.Module):
    """
    适用于飞机数据的小波神经网络威胁评估模型
    """

    def __init__(self):
        super(WNNAttackPlaneNet, self).__init__()

        # input layer
        self.fc1 = nn.Linear(6, 32)

        # 定义可训练参数a和b
        self.a = nn.Parameter(torch.Tensor(6))
        self.b = nn.Parameter(torch.Tensor(6))
        # 正态分布初始化参数a和b
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)
        self.relu = nn.ReLU()
        # output layer
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = (x - self.b) / self.a
        x = wavelet.apply(x)
        # pass through input layer
        x = self.fc1(x)

        x = self.relu(x)
        # pass through output layer
        x = self.fc3(x)

        return x


class WNNSupportPlaneNet(nn.Module):
    """
    适用于飞机数据的小波神经网络威胁评估模型
    """

    def __init__(self):
        super(WNNSupportPlaneNet, self).__init__()

        # input layer
        self.fc1 = nn.Linear(6, 32)

        # 定义可训练参数a和b
        self.a = nn.Parameter(torch.Tensor(6))
        self.b = nn.Parameter(torch.Tensor(6))
        # 正态分布初始化参数a和b
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)
        self.relu = nn.ReLU()
        # output layer
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = (x - self.b) / self.a
        x = wavelet.apply(x)
        # pass through input layer
        x = self.fc1(x)

        x = self.relu(x)
        # pass through output layer
        x = self.fc3(x)

        return x


class WNN_TA(object):
    """
    与墨子平台衔接的小波神经网络威胁评估算法
    """

    def __init__(self, targetData: list):
        # 各目标的所有可得信息，用于威胁评估、
        # DDDD：保卫要地重要性、毁伤概率、速度、射程、飞临时间、高度、突防能力
        # 其他DD：保卫要地重要性、毁伤概率、速度、RCS、飞临时间、高度、航路捷径、机动能力
        # 作战FJ：保卫要地重要性、飞机对地攻击作战效能、速度、RCS、高度、航路捷径
        # 作战支援FJ：保卫要地重要性、对地支援作战效能、速度、RCS、高度、距离

        super(WNN_TA, self).__init__()
        self.DataList = targetData

    def run(self):
        """
        逐次处理单个目标的威胁评估信息
        :return:
        """
        threat = []
        predicted = 0.0

        DnnDDDD = WNNDDDDNet().to(device)
        DnnDDDD.load_state_dict(
            torch.load(r"WNN\WNN_DDDD_model.pth", map_location=device), strict=False)  # pytoch 导入模型
        DnnDDDD.eval()  # 这里指评价模型，不反传，所以用eval模式

        DnnMissile = WNNMissileNet().to(device)
        DnnMissile.load_state_dict(
            torch.load(r"WNN\WNN_missile_model.pth", map_location=device), strict=False)  # pytoch 导入模型
        DnnMissile.eval()  # 这里指评价模型，不反传，所以用eval模式

        DnnAttackPlane = WNNAttackPlaneNet().to(device)
        DnnAttackPlane.load_state_dict(
            torch.load(r"WNN\WNN_attackPlane_model.pth", map_location=device), strict=False)  # pytoch 导入模型
        DnnAttackPlane.eval()  # 这里指评价模型，不反传，所以用eval模式

        DnnSupportPlane = WNNSupportPlaneNet().to(device)
        DnnSupportPlane.load_state_dict(
            torch.load(r"WNN\WNN_supportPlane_model.pth", map_location=device), strict=False)  # pytoch 导入模型
        DnnSupportPlane.eval()  # 这里指评价模型，不反传，所以用eval模式

        missileInfo = []
        planeInfo = []
        targetFlag = []
        Damage_list = []
        Base_list = []
        for singleData in self.DataList:
            print(singleData[0])
            if "弹道导弹" in singleData[0] or "核弹" in singleData[0]:
                singleThreat, singleInfo, damage, baseIndex = processSingleDDDDData(singleData)
                # missileInfo.append(singleInfo)
                targetFlag.append(0)
                plane = torch.from_numpy(singleThreat).to(device=device, dtype=torch.float32)
                DDDDThreatPredicted = DnnDDDD(plane)
                predicted = DDDDThreatPredicted.detach().cpu().item()  # 输出结果torch tensor，需要转化为numpy类型来进行可视化
            elif ("导弹" in singleData[0] and singleData[0].find("导弹") > 0) or "炸弹" in singleData[0]:
                singleThreat, singleInfo, damage, baseIndex = processSingleMissileData(singleData)
                # print("a")
                # missileInfo.append(singleInfo)
                targetFlag.append(0)
                missile = torch.from_numpy(singleThreat).to(device=device, dtype=torch.float32)
                missileThreatPredicted = DnnMissile(missile)
                predicted = missileThreatPredicted.detach().cpu().item()  # 输出结果torch tensor，需要转化为numpy类型来进行可视化
                # 根据类别进行加权
                if "导弹" in singleData[0]:
                    predicted = predicted * 0.6
                else:
                    predicted = predicted * 0.5

            elif "战斗机" in singleData[0] or "直升机" in singleData[0] or "轰炸机" in singleData[0] or "女武神无人机" in singleData[0] or "诡骗丽影无人战斗机" in singleData[0]:
                singleThreat, singleInfo, damage, baseIndex = processSinglePlaneData(singleData)
                planeInfo.append(singleInfo)
                targetFlag.append(1)
                plane = torch.from_numpy(singleThreat).to(device=device, dtype=torch.float32)
                planeThreatPredicted = DnnAttackPlane(plane)
                predicted = planeThreatPredicted.detach().cpu().item()  # 输出结果torch tensor，需要转化为numpy类型来进行可视化
                # 根据类别进行加权
                if "战斗机" in singleData[0] or "女武神无人机" in singleData[0] or "诡骗丽影无人战斗机" in singleData[0]:
                    predicted = predicted * 0.9
                elif "直升机" in singleData[0]:
                    predicted = predicted * 0.6
                else:
                    predicted = predicted * 0.9
            elif "干扰机" in singleData[0] or "侦察机" in singleData[0] or "预警机" in singleData[0]:
                singleThreat, singleInfo, damage, baseIndex = processSingleSupportPlaneData(singleData)
                planeInfo.append(singleInfo)
                targetFlag.append(1)
                plane = torch.from_numpy(singleThreat).to(device=device, dtype=torch.float32)
                planeThreatPredicted = DnnSupportPlane(plane)
                predicted = planeThreatPredicted.detach().cpu().item()  # 输出结果torch tensor，需要转化为numpy类型来进行可视化
                # 根据类别进行加权
                if "干扰机" in singleData[0]:
                    predicted = predicted * 0.8
                elif "侦查机" in singleData[0]:
                    predicted = predicted * 0.7
                else:
                    predicted = predicted * 0.8
            else:
                predicted = 0.2
                damage = 0.2
                baseIndex = 1
            # print(f'Damage_list:',Damage_list)
            # print(f'threat:', threat)
            # print(f'Base_list:', Base_list)
            Damage_list.append(damage)
            threat.append(predicted)
            Base_list.append(baseIndex)

        return threat, Damage_list, Base_list
