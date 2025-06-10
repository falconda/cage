import torch
import torch.nn as nn
import numpy as np
import sys
import copy
import csv
sys.path.append('D:\\workplace\\moziai\\mozi_ai_sdk\\FKFD\\WTA')

# from Models import LSTMModel
from Models import MLPModel
# from Models import RnnModel


# 目标：用目标函数去让神经网络得到损失进行收敛
class ExpectLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        loss_mat = x*z
        loss_mat1 = 1 - loss_mat
        loss_1 = torch.sum(loss_mat1)
        loss_mat2 = y*z
        loss_2 = torch.sum(loss_mat2)
        loss = 0.5 * loss_1 + 0.5 * loss_2
        # loss = loss_1
        # loss_1 基地剩余最大化
        # loss_2 消灭威胁最大化
        return -loss


class MLP_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.threat_value = Wei         # [1, num_target]
        self.hit_probability = Pij      # [num_weapon, num_target]
        self.feasibility = Fij          # [num_weapon, num_target]
        self.damage_probability = Qjk   # [num_target, num_base],num_base=3，并且每一行只有一个元素不为0
        self.asset_value = V_a           # [1, num_base]， num_base=3
        self.asset_value_sum = sum(V_a[0])
        self.threat_value_sum = sum(Wei[0])
        self.Tar_to_Asset = np.nonzero(self.damage_probability)[1]
        self.damage_probability_sum = np.zeros((1, self.num_target))
        for i in range(self.num_target):
            self.damage_probability_sum[0][i] = sum(self.damage_probability[i, :])

        # 使用GPU训练
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        # print('torch.cuda.is_available: ', torch.cuda.is_available())

        # 网络参数设定
        self.input_size = num_weapon * num_target
        self.hidden_size = 2 * num_weapon * num_target
        self.output_size = num_weapon * num_target
        self.batch_size = 1
        self.num_layers = 1
        self.iter_max = 30

        self.pointer1 = False   # 打印两个指标
        self.pointer2 = False    # 保存迭代文件

        # MLP MLPModel
        self.decoder_net = MLPModel(self.input_size, self.hidden_size, self.output_size)
        self.decoder_optim = torch.optim.Adam(self.decoder_net.parameters(), lr=0.01)
        self.lossf = ExpectLoss()

        self.decoder_net = self.decoder_net.to(self.device)
        self.list_1 = np.zeros((self.num_weapon, self.num_target))
        self.best_plan = np.full((self.num_weapon), -1)

        self.alpha = 1  # 用于打击概率:越大打击概率占的比重越大
        self.beta = 1   # 用于威胁值的权重
        self.gamma = 1  # 用于损伤概率

        # 构造损失输入
        temp1 = 1 - self.hit_probability
        temp2 = np.tile(self.damage_probability_sum, (num_weapon, 1))
        temp3 = np.tile(self.threat_value, (num_weapon, 1))
        expect_1 = temp1 * temp2                    # 基地损失最小化
        expect_2 = self.hit_probability * temp3     # 消灭威胁最大化
        self.expect_1 = torch.tensor(expect_1, dtype=torch.float).to(self.device)
        self.expect_2 = torch.tensor(expect_2, dtype=torch.float).to(self.device)

        expect_temp1 = (1-self.hit_probability)
        damage_expect_arr = (self.beta * self.threat_value) * (self.gamma * self.damage_probability_sum)
        damage_expect = np.tile(damage_expect_arr, (num_weapon, 1))
        self.expect = torch.tensor(expect_temp1 * damage_expect, dtype=torch.float).to(self.device)
        self.inputs = self.expect.view(1, 1, -1)
        self.inputs = self.inputs.to(self.device)

        # 转换为tensor
        self.threat_value = torch.tensor(self.threat_value, dtype=torch.float)
        self.hit_probability = torch.tensor(self.hit_probability, dtype=torch.float)
        self.feasibility = torch.tensor(self.feasibility, dtype=torch.float)
        self.damage_probability = torch.tensor(self.damage_probability, dtype=torch.float)
        self.damage_probability_sum = torch.tensor(self.damage_probability_sum, dtype=torch.float)
        self.list_1 = np.zeros((self.num_weapon, self.num_target)) # 保存最后输出用

        # 先移动到GPU
        self.threat_value = self.threat_value.to(self.device)
        self.hit_probability = self.hit_probability.to(self.device)
        self.feasibility = self.feasibility.to(self.device)
        self.damage_probability = self.damage_probability.to(self.device)
        self.damage_probability_sum = self.damage_probability_sum.to(self.device)
        # self.base_value = self.base_value.to(self.device)
        self.hit_probability_cpu = self.hit_probability.clone()  # 拷贝一份到cpu上
        # self.hit_probability_cpu = self.hit_probability_cpu.cpu()

    def cal_iter(self, single_plan):
        single_plan = single_plan.astype(int)
        asset_value = copy.deepcopy(self.asset_value)
        value = np.zeros((1, self.num_weapon))
        for m in range(len(single_plan)):
            # m:武器编号  single_plan[m]：目标编号
            if single_plan[m] != -1:
                # 最大化剩余基地价值
                # 拦截概率转换为突防概率
                p1 = (1 - self.hit_probability[m][single_plan[m]])
                # 突防概率*损伤概率
                p2 = p1 * self.damage_probability_sum[0][single_plan[m]]
                # 对应打击基地确定：0/1/2
                target_base = self.Tar_to_Asset[single_plan[m]]
                # 对应造成基地损失，得到剩余基地价值
                asset_value[0][target_base] = asset_value[0][target_base] * (1 - p2)

        a = np.array(asset_value).flatten().tolist()    # 转为只有一个中括号的
        value_base_remain = sum(a)
        value_threat_sum = np.sum(value)
        per1 = value_base_remain / self.asset_value_sum
        # per1 = per1.detach().cpu().numpy()
        per2 = value_threat_sum / self.threat_value_sum
        return per1, per2

    def mat_to_plan(self, matrix):
        plan = np.full((self.num_weapon), -1)
        for k in range(self.num_weapon):
            max_index1 = np.argmax(matrix)
            max_row1, max_col1 = np.unravel_index(max_index1, matrix.shape)
            plan[max_row1] = max_col1
            matrix[max_row1, :] = -1
            matrix[:, max_col1] = -1
        return plan

    def train(self):
        for epoch in range(self.iter_max):
            # 优化器梯度归零
            self.decoder_optim.zero_grad()
            outputs_arr = self.decoder_net(self.inputs)
            # 对输出进行操作:转换为num_weapon x num_target
            outputs = outputs_arr.view(self.num_weapon, self.num_target)
            outputs = outputs.to(self.device)
            # print(outputs_save.shape)
            loss1 = self.lossf(self.expect_1, self.expect_2, outputs)
            # print('outputs_save:{}{}'.format(outputs_save.shape, outputs_save.dtype))
            # print('plan_sup:{}{}'.format(self.expect.shape, self.expect.dtype))
            # print('loss of this iteration:{}'.format(loss1))
            loss1.backward()
            self.decoder_optim.step()

            iter_mar1 = outputs.clone().detach()
            iter_mar2 = iter_mar1.cpu()
            iter_mar = iter_mar2.numpy()
            iter_plan = self.mat_to_plan(iter_mar)
            # print('MLP_WTA')
            # print('本次迭代方案为：{}'.format(iter_plan))
            p1, p2 = self.cal_iter(iter_plan)
            if self.pointer1:
                print('剩余基地价值比例为：{}'.format(p1))
                print('预计打击目标比例为：{}'.format(p2))
            # 保存数据到csv文件
            if self.pointer2:
                with open('data_MLP_0.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([p1, p2])

        outputs_list = outputs.clone().detach()
        self.list_1 = outputs_list.cpu()

    def run(self):
        self.train()
        list_11 = torch.nn.functional.softmax(self.list_1, dim=1)
        list_2 = list_11.numpy()

        # print(list_2.shape)
        # 依据武器进行分配
        self.best_plan = self.mat_to_plan(list_2)
        self.best_plan = self.best_plan.astype('int32')
        self.best_plan = self.best_plan.tolist()
        # print(self.best_plan)
        # print("输出方案为{}".format(self.best_plan))
        return self.best_plan

