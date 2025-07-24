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
STATIC_SIZE = 7  # (x, y)
STATIC1_SIZE = 6
STATIC2_SIZE = 7
max_grad_norm = 2
actor_lr = 5e-4
critic_lr = 5e-4
file_number = 0
reward_line = []

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
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)  # 在第3维度与static_hidden保持一致
        hidden = torch.cat((static_hidden, hidden), 1)  # 第二个维度拼接
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)# 第三个维度概率归一化
        return attns

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, decoder_hidden, last_hh):
        # static_hidden.size() [256, 128, 20]
        # decoder_hidden.size() [256, 128, 1]

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)  # [256,1,128]
        rnn_out = rnn_out.squeeze(1)  # 只移除大小为1的维度[256,128]
        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an input context
        enc_attn = self.encoder_attn(static_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats) 2维3维置换

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

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


class PN_WTA_red:
    def __init__(self, step=0):
        """只负责加载模型，不绑定输入数据"""
        self.data_line = []
        self.data = []

        # ✅ 初始化 Actor & Critic
        self.actor_model = DRL4TSP(STATIC_SIZE,
                              STATIC1_SIZE,
                              STATIC2_SIZE,
                              256,
                              None,
                              wta_update_mask,
                              1,
                              0.1).to(device)

        self.critic_model = StateCritic(STATIC_SIZE, STATIC1_SIZE, STATIC2_SIZE, 256, 20, 20).to(device)

        # ✅ 加载模型权重（只加载一次）
        if step == 0:
            weight = torch.load(os.path.join(WTAModelPath, "actor.pt"), map_location='cpu')
            self.actor_model.load_state_dict(weight)
            self.actor_model.to(device)
        else:
            NEWModelPath = os.path.join(current_dir, "pointer_model_new")
            checkpoint_actor_path = os.path.join(NEWModelPath, "checkpoints_red", str(step-1), "actor.pt")
            weight = torch.load(checkpoint_actor_path, map_location='cpu')
            self.actor_model.load_state_dict(weight)
            self.actor_model.to(device)

            checkpoint_critic_path = os.path.join(NEWModelPath, "checkpoints_red", str(step-1), "critic.pt")
            weight = torch.load(checkpoint_critic_path, map_location='cpu')
            self.critic_model.load_state_dict(weight)
            self.critic_model.to(device)

    def init_WTA_input(self, num_weapon, num_target, vt, pij, fij,  qjk, vb):
        num_samples = 1
        # WTA信息处理
        Randint = len(vb)  # 基地的数量
        positions = []
        # 遍历 self.qjk 中的每个张量
        for tensor in qjk:
            # 使用 nonzero() 找到不为零的元素的位置
            nonzero_indices = torch.nonzero(tensor)
            # 选择每行的第一个非零元素的索引
            if nonzero_indices.size(0) > 0:
                # 获取第一个非零元素的位置
                first_nonzero_pos = nonzero_indices[0]
                positions.append(first_nonzero_pos)
        # 将位置合并为一个 (1, 2) 形状的张量
        self.T_to_A = torch.stack(positions).view(1, -1)
        # 基地的价值的标准化
        self.norm_value = vb / 100
        # 目标对应基地的频率， 基地的价值占比
        self.f, self.base = trans_norm(self.T_to_A, Randint, self.norm_value)
        ##################
        self.f1 = self.f.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        self.base1 = self.base.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        #####################
        # 提取每一行的不为零的元素
        target_pij = []
        for row in qjk:
            # 获取该行中不为零的元素
            non_zero_elements = row[row != 0]
            target_pij.append(non_zero_elements[0])
        # 将提取的非零元素合并为一个张量
        self.target_pij = torch.stack(target_pij).unsqueeze(0).unsqueeze(0)
        # 目标打击基地的概率
        self.target_pij1 = self.target_pij.repeat(1, 1, num_weapon)
        reshaped_pij = pij.view(-1)  # 这将把 (32, 2) 张量转换为一个 64 元素的一维张量
        self.weapon_pij = reshaped_pij.unsqueeze(0).unsqueeze(0)
        reshaped_fij = fij.view(-1)  # 这将把 (32, 2) 张量转换为一个 64 元素的一维张量
        self.weapon_fij = reshaped_fij.unsqueeze(0).unsqueeze(0)
        # 武器打击目标的概率
        # 目标的威胁值
        repeated_vt = vt.repeat(num_weapon)
        self.target_threat1 = repeated_vt.unsqueeze(0).unsqueeze(0)
        # WTA相关信息输入
        self.dataset1 = torch.cat((self.f1, self.base1, self.target_pij1, self.weapon_pij, self.weapon_fij,
                                   self.target_threat1, self.weapon_pij), dim=1)

    def run(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a, step):
        self.num_weapon = torch.tensor(num_weapon).to(device)
        self.num_target = torch.tensor(num_target).to(device)
        self.Threat = Wei[0]
        self.Pij = Pij
        self.Fij = Fij
        self.Qjk = Qjk
        self.V_a = V_a[0]
        self.data_line = []

        self.actor_model.train()
        self.critic_model.train()

        plan, tour_logp, critic_est = self.trans_model()
        result = plan.tolist()[0]
        fitnesss = F7(result, self.num_weapon, self.num_target, len(self.V_a), self.Threat, self.Pij, self.Fij,
                      self.Qjk, self.V_a)

        self.data_line.append(fitnesss)
        self.data_line.append(tour_logp)
        self.data_line.append(critic_est)
        self.data.append(self.data_line)

        return result

    def trans_model(self):
        # 提取WTA数据
        self.vt = torch.tensor(self.Threat)
        self.pij = torch.tensor(self.Pij)
        self.fij = torch.tensor(self.Fij)
        self.qjk = torch.tensor(self.Qjk)
        self.vb = torch.tensor(self.V_a)
        self.init_WTA_input(self.num_weapon, self.num_target, self.vt, self.pij, self.fij, self.qjk, self.vb)
        static = self.dataset1.to(device)
        tour_indices, tour_logp = self.actor_model(self.num_weapon, self.num_target, static)
        plan = trans_to_plan(tour_indices, self.num_weapon, self.num_target)
        critic_est = self.critic_model(static).view(-1)

        return plan, tour_logp, critic_est


    def train_pointer(self, reward1, reward2):
        data = self.data
        actor_optim = optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        critic_optim = optim.Adam(self.critic_model.parameters(), lr=critic_lr)
        step = 0
        reward_mean = np.mean([x[0].item() for x in data])
        reward_line.append(reward_mean)

        save_dir = os.path.join(os.getcwd(), 'pointer_model_new')
        checkpoint_dir = os.path.join(save_dir, 'checkpoints_red')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

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
            reward = fitnesss + 0.5 * (1 - reward1 + reward2)
            # f = 1 / reward
            f = reward
            advantage = (f - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)
            all_rewards.append(f)

            # # 梯度清零、反向传播、优化器更新
            # actor_optim.zero_grad()
            # actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_grad_norm)  # 梯度裁剪防止爆炸
            # actor_optim.step()
            #
            # critic_optim.zero_grad()
            # critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_grad_norm)
            # critic_optim.step()

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

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(self.actor_model.state_dict(), save_path)
        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(self.critic_model.state_dict(), save_path)
        print('finish!')


def wta_update_mask(mask, chosen_idx, num_weapon=20, num_target=20):
    weapon_index = torch.div(chosen_idx, num_target, rounding_mode='floor')
    start_a = torch.mul(weapon_index, num_target)
    end_a = torch.mul(torch.add(weapon_index, 1), num_target)
    weapon_array = []
    for i in range(chosen_idx.size()[0]):
        weapon_array.append(list(range(int(start_a[i]), int(end_a[i]), 1)))
    weapon_mask = torch.tensor(weapon_array).to(device)
    target_array = []
    target_index = torch.remainder(chosen_idx, num_target)
    end_c = torch.add(target_index, num_weapon * num_target)
    for i in range(chosen_idx.size()[0]):
        target_array.append(list(range(target_index[i], end_c[i], num_target)))
    target_mask = torch.tensor(target_array).to(device)
    mask.scatter_(1, weapon_mask, 0)
    mask.scatter_(1, target_mask, 0)
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

def trans_norm(tensor_input, num_base, base_value):
    """
    input:
    tensor_input: n数量个目标打击基地的选择
    num_base: 基地数量
    base_value:基地价值的标准化
    output:
    random_norm:标准化基地选择，利用出现频率代替
    random_norm_base:目标打击基地价值的标准化

    """
    B = []
    C = []
    # num_sample
    row = int(tensor_input.size(0))
    # target
    column = int(tensor_input.size(1))
    random_norm = torch.zeros((row, column))
    random_norm_base = torch.zeros((row, column))
    for i in range(row):
        for j in range(num_base):
            count = torch.sum(torch.eq(tensor_input[i, :], j).int())
            B.append(int(count))
        C.append(B)
        B = []
    for i in range(row):
        for j in range(column):
            index = int(tensor_input[i][j])
            random_norm[i][j] = 1 / C[i][index]
            random_norm_base[i][j] = base_value[index]
    return random_norm, random_norm_base

def F7(x, num_w, num_t, num_b, vj, pij, fij, qij, wb):
    x0 = [-1] * num_w
    for i in range(num_w):
        x0[i] = x[i]
    J = 0
    target_used = [0] * 100
    all_bv = 0
    vj_total = 0
    remain_bv = [0] * 100
    base_remain = 0
    for b in range(num_b):
        remain_bv[b] = wb[b]
        all_bv += wb[b]
    for i in range(num_w):
        if x0[i] != -1:
            target = int(x0[i])
            target_used[target] = 1
            for b in range(num_b):
                remain_bv[b] -= (1 / num_b) * wb[b] * qij[target][b] * (1 - pij[i][target])
                if remain_bv[b] < 0:
                    remain_bv[b] = 0
    for j in range(num_t):
        vj_total += vj[j]
        if target_used[j] == 0:  # 若该目标没有被选择
            for b in range(num_b):
                remain_bv[b] -= (1 / num_b) * wb[b] * qij[j][b]
                if remain_bv[b] < 0:
                    remain_bv[b] = 0
    for b in range(num_b):
        base_remain += remain_bv[b]
    J2 = base_remain / all_bv  # 基地保留比例
    for i in range(num_w):
        if x0[i] != -1:
            t = int(x0[i])
            J = J + pij[i][t] * vj[t]
    J1 = J / vj_total  # 目标威胁消除比例

    return J1 * 0.5 + J2 * 0.5