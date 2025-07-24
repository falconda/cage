import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
current_dir = os.path.dirname(__file__)
WTAModelPath = os.path.join(current_dir, "model")


# 指针网络相关参数
STATIC_SIZE = 4  # (x, y)
STATIC1_SIZE = 6
STATIC2_SIZE = 7
max_grad_norm = 2
actor_lr = 5e-4
critic_lr = 5e-4
file_number = 0

class StateCritic(nn.Module):
    def __init__(self, static_size, static1_size, static2_size, hidden_size, num_weapon, num_target):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.static1_encoder = Encoder(static1_size, hidden_size)
        self.static2_encoder = Encoder(static2_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size, num_weapon * num_target, kernel_size=1)
        self.fc2 = nn.Conv1d(num_weapon * num_target, min(num_target, num_weapon), kernel_size=1)
        self.fc3 = nn.Conv1d(min(num_target, num_weapon), 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static):

        # Use the probabilities of visiting each
        # static_hidden = self.static_encoder(static) + self.static1_encoder(static1) + self.static2_encoder(static2)
        static = static.float()
        static_hidden = self.static_encoder(static)
        output = F.relu(self.fc1(static_hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Encoder(nn.Module):
    """使用一维卷积对静态和动态状态进行编码"""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input_data):
        return self.conv(input_data)

class Attention(nn.Module):
    """根据当前状态计算对输入节点的注意力 (简化版，匹配旧模型)"""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))
        # 注意这里的W尺寸是 2 * hidden_size，对应错误信息里的512
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, hidden), 1)

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns


class Pointer(nn.Module):
    """根据前一状态和输入嵌入计算下一状态 (修改版，匹配旧模型)"""

    def __init__(self, hidden_size, num_layers=1, dropout=0.):
        super(Pointer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), device=device, requires_grad=True))
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        # 注意这里的 Attention 指向上面那个简化版
        self.encoder_attn = Attention(hidden_size)
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, decoder_hidden, last_hh):
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            last_hh = self.drop_hh(last_hh)

        # 注意这里的调用，没有 dynamic_hidden
        enc_attn = self.encoder_attn(static_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)
        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)
        energy_safe = energy.clone()
        W_safe = W.clone()
        v_safe = v.clone()

        probs = torch.bmm(v_safe, torch.tanh(torch.bmm(W_safe, energy_safe))).squeeze(1)

        # probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        return probs, last_hh

class DRL4TSP(nn.Module):
    def __init__(self, static_size, static1_size, static2_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.1):
        super(DRL4TSP, self).__init__()

        self.static_size = static_size
        self.static1_size = static1_size
        self.update_fn = update_fn
        self.mask_fn = mask_fn
        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.static1_encoder = Encoder(static1_size, hidden_size)
        self.static2_encoder = Encoder(static2_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.decoder1 = Encoder(static1_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)
        self.x1 = torch.zeros((1, static1_size, 1), requires_grad=True, device=device)

    def forward(self, num_weapon, num_target, static, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        self.num_target = num_target
        self.num_weapon = num_weapon

        batch_size, input_size, sequence_size = static.size()
        decoder1_input = None
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, self.static_size, 1)
            decoder1_input = self.x1.expand(batch_size, self.static1_size, 1)
        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)  # 初始化 mask 为全 1

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        # max_steps = sequence_size if self.mask_fn is None else 1000
        max_steps = min(self.num_weapon, self.num_target)
        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        # static_hidden = self.static_encoder(static) + self.static1_encoder(static1) + self.static2_encoder(static2)
        static = static.float()
        static_hidden = self.static_encoder(static)

        for i in range(max_steps):
            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            # decoder_hidden = self.decoder(decoder_input) + self.decoder1(decoder1_input)
            decoder_hidden = self.decoder(decoder_input)
            probs, last_hh = self.pointer(static_hidden, decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)  # [256,20]

            # 避免 NaN 和全 0 行
            probs = torch.nan_to_num(probs, nan=1e-6)  # 替换 NaN
            probs[probs.sum(dim=1) == 0] = 1 / probs.size(1)  # 处理全 0 行，设为均匀分布
            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)
                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, ptr.data, self.num_weapon, self.num_target).detach()
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            # 从索引找到坐标
            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()
        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp


class PN_WTA_blue:
    def __init__(self, step=0):
        """只负责加载模型，不绑定输入数据"""
        self.data_line = []
        self.data = []

        # ✅ 初始化 Actor & Critic
        self.actor_model = DRL4TSP(
            STATIC_SIZE, STATIC1_SIZE, STATIC2_SIZE,
            256, None, wta_update_mask, 1, 0.1
        ).to(device)

        self.critic_model = StateCritic(
            STATIC_SIZE, STATIC1_SIZE, STATIC2_SIZE,
            256, 20, 20
        ).to(device)

        # ✅ 加载模型权重（只加载一次）
        if step == 0:
            weight = torch.load(os.path.join(WTAModelPath, "actor_blue.pt"), map_location='cpu')
            self.actor_model.load_state_dict(weight)
            self.actor_model.to(device)
        else:
            NEWModelPath = os.path.join(current_dir, "pointer_model_new")
            checkpoint_actor_path = os.path.join(NEWModelPath, "checkpoints_blue", str(step-1), "actor_blue.pt")
            weight = torch.load(checkpoint_actor_path, map_location='cpu')
            self.actor_model.load_state_dict(weight)
            self.actor_model.to(device)

            checkpoint_critic_path = os.path.join(NEWModelPath, "checkpoints_blue", str(step-1), "critic_blue.pt")
            weight = torch.load(checkpoint_critic_path, map_location='cpu')
            self.critic_model.load_state_dict(weight)
            self.critic_model.to(device)

    def plan_to_matrix(self, plan: np.ndarray, num_weapon, num_target) -> np.ndarray:
        """根据 plan 映射出单位-目标 0/1 矩阵"""
        x = np.zeros((num_weapon, num_target), dtype=int)
        used_targets = set()
        for i in range(num_weapon):
            j = plan[i]
            if j == -1 or j in used_targets:
                continue
            x[i][j] = 1
            used_targets.add(j)
        return x

    def init_WTA_input(self, pij_t, wj_t, ava_weapon_t, tij_t, num_weapon, num_target):
        """构造 WTA 输入张量（动态维度）"""
        repeated_vt = wj_t.repeat(num_weapon)
        repeated_vt_arr = repeated_vt.view(-1)
        target_threat1 = repeated_vt_arr.unsqueeze(0).unsqueeze(0)

        pij1_arr = pij_t.view(-1)
        pij1 = pij1_arr.unsqueeze(0).unsqueeze(0)

        tij1_arr = tij_t.view(-1)
        tij1 = tij1_arr.unsqueeze(0).unsqueeze(0)

        ava_weapon1 = ava_weapon_t.repeat_interleave(num_target)
        ava_weapon1 = ava_weapon1.unsqueeze(0).unsqueeze(0)

        return torch.cat((target_threat1, pij1, tij1, ava_weapon1), dim=1)

    def get_actual_assignment(self, x: np.ndarray, tij, ava_weapon) -> np.ndarray:
        """根据 0/1 配对矩阵生成实际武器分配矩阵"""
        num_weapon, num_target = x.shape
        x_actual = np.zeros_like(x, dtype=float)
        for i in range(num_weapon):
            for j in range(num_target):
                if x[i][j] == 1:
                    x_actual[i][j] = min(tij[i][j], ava_weapon[i])
        return x_actual

    def fitness(self, x_plan, pij, wj, tij):
        """根据计划矩阵计算适应度"""
        num_weapon, num_target = pij.shape
        x_actual = self.get_actual_assignment(x_plan, tij, np.sum(x_plan, axis=1))  # or pass ava_weapon separately
        total_value = 0.0
        for j in range(num_target):
            destroy_ratio = 0.0
            for i in range(num_weapon):
                if tij[i, j] > 0:
                    destroy_ratio += pij[i, j] * x_actual[i, j] / tij[i, j]
            total_value += wj[j] * min(1.0, destroy_ratio)
        return total_value

    def run(self, pij, wj, weapon_num, tij):
        """
         动态推理：支持不同轮次传不同维度的输入
        pij: (num_weapon, num_target)
        wj:  (num_target,)
        weapon_num: (num_weapon,)
        tij: (num_weapon, num_target)
        """
        #  动态维度
        pij = np.array(pij)
        wj = np.array(wj)
        ava_weapon = np.array(weapon_num)
        tij = np.array(tij)
        num_weapon, num_target = pij.shape
        self.data_line = []

        #  转换为 Tensor
        pij_t = torch.tensor(pij, dtype=torch.float32)
        wj_t = torch.tensor(wj, dtype=torch.float32)
        ava_weapon_t = torch.tensor(ava_weapon, dtype=torch.float32)
        tij_t = torch.tensor(tij, dtype=torch.float32)

        #  构造模型输入
        dataset1 = self.init_WTA_input(pij_t, wj_t, ava_weapon_t, tij_t, num_weapon, num_target)
        static = dataset1.to(device)

        #  模型推理
        tour_indices, tour_logp = self.actor_model(num_weapon, num_target, static)
        plan = trans_to_plan(tour_indices, num_weapon, num_target)
        critic_est = self.critic_model(static).view(-1)

        #  计划矩阵 + 适应度
        result = plan.tolist()[0]
        plan_mat = self.plan_to_matrix(result, num_weapon, num_target)
        fitnesss = self.fitness(plan_mat, pij, wj, tij)

        #  保存数据用于训练
        data_line = [fitnesss, tour_logp, critic_est]
        self.data_line.append(fitnesss)
        self.data_line.append(tour_logp)
        self.data_line.append(critic_est)
        self.data.append(self.data_line)
        return plan_mat, result, [data_line]


    def train_pointer(self, reward1, reward2, record_data):
        actor_optim = optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        critic_optim = optim.Adam(self.critic_model.parameters(), lr=critic_lr)
        step = 0

        save_dir = os.path.join(os.getcwd(), 'pointer_model_new')
        checkpoint_dir = os.path.join(save_dir, 'checkpoints_blue')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        data = self.data
        all_actor_losses = []
        all_critic_losses = []
        all_rewards = []

        for data_line in data:
            print(step)
            step+=1
            # data_line是二维列表
            fitnesss = data_line[0]
            tour_logp = data_line[1]
            critic_est = data_line[2]

            torch.tensor(fitnesss, dtype=torch.float32, device=device)
            reward = fitnesss + 0.5 * (reward1 + reward2)
            # f = 1 / reward
            f = reward
            advantage = (f - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)
            all_rewards.append(f)

            # 梯度清零、反向传播、优化器更新
            # actor_optim.zero_grad()
            # actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_grad_norm)  # 梯度裁剪防止爆炸
            # actor_optim.step()
            #
            # critic_optim.zero_grad()
            # critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_grad_norm)
            # critic_optim.step()
            # 对所有样本求平均

        mean_actor_loss = torch.stack(all_actor_losses).mean()
        mean_critic_loss = torch.stack(all_critic_losses).mean()
        # mean_reward = torch.stack(all_rewards).mean()  # 平均损伤

        # 一次性更新网络
        actor_optim.zero_grad()
        mean_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_grad_norm)
        actor_optim.step()

        critic_optim.zero_grad()
        mean_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_grad_norm)
        critic_optim.step()
        step += 1
        self.data.clear()

        # Save the weights
        global file_number
        epoch_dir = os.path.join(checkpoint_dir, '%s' % file_number)
        file_number += 1
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor_blue.pt')
        torch.save(self.actor_model.state_dict(), save_path)
        save_path = os.path.join(epoch_dir, 'critic_blue.pt')
        torch.save(self.critic_model.state_dict(), save_path)
        print('finish!')


def wta_update_mask(mask, chosen_idx, num_weapon, num_target):
    """更新掩码，确保每个武器和目标只被选择一次"""
    weapon_index = torch.div(chosen_idx, num_target, rounding_mode='floor')
    target_index = torch.remainder(chosen_idx, num_target)

    # 屏蔽被选中的武器行
    start_w = torch.mul(weapon_index, num_target)
    weapon_mask_indices = torch.arange(num_target, device=device).unsqueeze(0).repeat(chosen_idx.size(0),
                                                                                      1) + start_w.unsqueeze(1)
    mask.scatter_(1, weapon_mask_indices, 0)

    # 屏蔽被选中的目标列
    target_mask_indices = torch.arange(0, num_weapon * num_target, num_target, device=device).unsqueeze(
        0) + target_index.unsqueeze(1)
    mask.scatter_(1, target_mask_indices, 0)

    return mask

def trans_to_plan(index, num_weapon, num_target):
    batch_size = index.size(0)

    # 新建一个固定形状的 tensor：全部初始化为 -1，代表未分配
    index_sort = torch.full((batch_size, num_weapon), -1, dtype=torch.long, device=index.device)

    for row, v in enumerate(index):
        temp_sort, _ = torch.sort(index[row])

        # 计算武器编号和目标编号
        weapon_index = torch.div(temp_sort, num_target, rounding_mode='floor').tolist()
        target_index = torch.remainder(temp_sort, num_target).tolist()

        # 初始化分配方案
        plan = [-1] * num_weapon
        for k, w in zip(weapon_index, target_index):
            if 0 <= k < num_weapon:
                plan[k] = w  # 第 k 个武器打第 w 个目标

        # 替换当前行
        index_sort[row] = torch.tensor(plan, device=index.device)

    return index_sort
