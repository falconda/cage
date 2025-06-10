import random
import numpy as np
import timeit
import torch
import sys
sys.path.append('D:\\workplace\\moziai\\mozi_ai_sdk\\FKFD\\WTA')


from HDGWO_WTA import HDGWO_WTA
from IGWO_WTA import IGWO_WTA
from MBO_WTA import MBO_WTA
from DDEMRR_WTA import DDEMRR_WTA
from DWHO_WTA import DWHO_WTA
from FSA_WTA import FSA_WTA
from GRO_WTA import GRO_WTA
from RIME_WTA import RIME_WTA
from CSA_WTA import CSA_WTA
from SABO_WTA import SABO_WTA
from ISA_WTA import ISA_WTA
from POA_WTA import POA_WTA
from SPO_WTA import SPO_WTA

from RNN_WTA import RNN_WTA
# from LSTM_WTA_1 import LSTM_WTA_1
# from LSTM_WTA_2 import LSTM_WTA_2
from MLP_WTA import MLP_WTA
from LSTM_WTA import LSTM_WTA
from VIA_WTA import VIA_WTA
from PIA_WTA import PIA_WTA

from LAMGC_WTA import LAMGC_WTA

from DF_WTA import DF_WTA

# AL = [HDGWO_WTA, IGWO_WTA, MBO_WTA, DDEMRR_WTA, DWHO_WTA, FSA_WTA, GRO_WTA, RIME_WTA, CSA_WTA, SABO_WTA, ISA_WTA, POA_WTA, SPO_WTA, RNN_WTA, MLP_WTA, LSTM_WTA, VIA_WTA, PIA_WTA, LAMGC_WTA]
# AL_name = ['HDGWO_WTA', 'IGWO_WTA', 'MBO_WTA', 'DDEMRR_WTA', 'DWHO_WTA', 'FSA_WTA', 'GRO_WTA', 'RIME_WTA', 'CSA_WTA', 'SABO_WTA', 'ISA_WTA', 'POA_WTA', 'SPO_WTA', 'RNN_WTA', 'MLP_WTA', 'LSTM_WTA', 'VIA_WTA', 'PIA_WTA', 'LAMGC_WTA']
# AL = [MLP_WTA, MLP_WTA_1, MLP_WTA_2]
# AL_name = ['RNN_WTA', 'RNN_WTA_1', 'RNN_WTA_2']
# AL = [LSTM_WTA, LSTM_WTA_1, LSTM_WTA_2]
# AL_name = ['RNN_WTA', 'RNN_WTA_1', 'RNN_WTA_2']
AL = [DF_WTA]
AL_name = ['DF_WTA']
AL_time = {}
t = 0
num_weapon = 5
num_target = 5
iter_max = 1
np.random.seed(123)
Mar_pij = np.random.rand(num_weapon, num_target)
Vec_Wei = np.random.rand(1, num_target)
Fij = np.ones((num_weapon, num_target))
Fij = np.random.randint(2, size=(num_weapon, num_target))
V_a = [[80, 85, 90]]
# 生成目标打击基地的概率矩阵
T_to_A = np.random.randint(0, 2, [1, num_target])[0]
qjk = np.zeros((num_target, 3))

for k, v in enumerate(T_to_A):
    qjk[k][v] = random.random()
print(qjk)
Tar_to_Asset = np.nonzero(qjk)[1]
for key, A in enumerate(AL):
    for n in range(iter_max):
        start = timeit.default_timer()
        model = A(num_weapon, num_target, Vec_Wei, Mar_pij, Fij, qjk, V_a)
        best_plan = model.run()
        print('最终输出方案为{}'.format(best_plan))
        end = timeit.default_timer()
        print('Running time: %s Seconds' % (end - start))
        t = t + (end - start)
    average_t = t / iter_max
    AL_time[AL_name[key]] = average_t
    t = 0
print(AL_time)