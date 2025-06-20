import os.path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
# 拼接路径，指向当前文件夹下的 "model" 文件夹
WTAModelPath = os.path.join(current_dir, "model")

device = torch.device('cpu')

print(f"Using device: {device}")


class PN_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.Threat = Wei[0]
        self.Pij = Pij
        self.Fij = Fij
        self.Qjk = Qjk
        self.V_a = V_a[0]

    def trans_model(self, weapon, target, threat, pof, pod, value, model):
        threat1 = torch.tensor(threat)
        threat2 = threat1.repeat(1, 1, weapon)
        pij1 = torch.tensor(pof)
        pij2 = torch.reshape(pij1, (1, 1, weapon * target))
        value = [v / max(value) for v in value]
        target_hit, hit_base = list(np.nonzero(pod))[0], list(np.nonzero(pod))[1]
        pod_array = np.zeros(target)
        value_array = np.zeros(target)
        for k, v in zip(target_hit, hit_base):
            pod_array[k] = pod[k][v]
            value_array[k] = value[v]
        pod_tensor = torch.tensor(pod_array).repeat(1, 1, weapon)
        value_tensor = torch.tensor(value_array).repeat(1, 1, weapon)
        static_input = torch.cat((value_tensor, pod_tensor, threat2, pij2), dim=1).to(dtype=torch.float32)
        dynamic_input = torch.zeros((1, 1, self.num_weapon * self.num_target)).to(dtype=torch.float32)
        out = model(static_input.to(device), dynamic_input.to(device), None)
        temp = out[0][0]
        weapon_index = torch.div(temp, self.num_target, rounding_mode='floor').tolist()
        target_index = torch.remainder(temp, self.num_target).tolist()
        plan = [-1] * self.num_weapon
        for k, v in zip(weapon_index, target_index):
            plan[k] = v
        return plan

    def run(self):
        # 模型参数加载
        weight = torch.load(os.path.join(WTAModelPath, "pointnet.pt"), map_location='cpu')
        actor_model = DRL4WTA(4, 1, 256, self.num_weapon, self.num_target, None, wta_update_mask, 1, 0.1)
        actor_model.load_state_dict(weight)
        actor_model = actor_model.to(device)
        actor_model.eval()
        plan = self.trans_model(self.num_weapon, self.num_target, self.Threat, self.Pij, self.Qjk, self.V_a,
                                actor_model)

        return plan


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

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)  # 在第3维度与static_hidden保持一致
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)  # 第二个维度拼接
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

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
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
        # 计算decoder输出rnn_out与输入序列static_hidden, dynamic_hidden的相似度  size：(1,1,num_w*num_t)
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        # 矩阵乘法
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats) 2维3维置换

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        # 最终的注意力权重
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh


class DRL4WTA(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size, num_weapon, num_target,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.1):
        super(DRL4WTA, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.num_target = num_target
        self.num_weapon = num_weapon
        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)
        mask = torch.ones(batch_size, sequence_size, device=device)
        tour_idx, tour_logp = [], []
        max_steps = min(self.num_weapon, self.num_target)
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for i in range(max_steps):

            if not mask.byte().any():
                break
            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)  # [256,20]
            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()
            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)
            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data, self.num_weapon, self.num_target).detach()
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            # 从索引找到坐标
            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp


def wta_update_mask(mask, dynamic, chosen_idx, num_weapon=20, num_target=20):
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


weapon_num, target_num = 5, 3

# 生成随机数
Mar_pij = np.random.rand(weapon_num, target_num)
Vec_Wei = np.random.rand(1, target_num)
Fij = np.ones((weapon_num, target_num))
V_a = [[80, 85, 90]]
# 生成目标打击基地的概率矩阵
T_to_A = np.random.randint(0, len(V_a[0]) - 1, [1, target_num])[0]
qjk = np.zeros((target_num, len(V_a[0])))
for k, v in enumerate(T_to_A):
    qjk[k][v] = random.random()
plan = PN_WTA(weapon_num, target_num, Vec_Wei, Mar_pij, Fij, qjk, V_a).run()
print(plan)
