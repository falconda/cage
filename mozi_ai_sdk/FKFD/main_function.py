import traceback
import logging
import re
import os
import argparse
import csv
import timeit
from datetime import datetime
from openpyxl import load_workbook
import sys
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mozi_ai_sdk.FKFD import dataProcess
from mozi_ai_sdk.FKFD.WNN import WNN_TA

from mozi_ai_sdk.FKFD.env.env import Environment
from mozi_ai_sdk.FKFD.env import etc
from mozi_ai_sdk.FKFD.functions_red import feasibility, probability_of_hit, get_target_A, get_weapon_set, \
    get_current_num, weapon_info, get_class_num, transpose
from mozi_ai_sdk.FKFD.functions_blue import (monitor_attack_results, monitor_aircraft_damage, pij_generate, evaluate_targets,
                                             extract_targets_attributes, extract_target_encoded_attributes, apply_assignment_with_limits, build_tij_matrix, get_red_damage)
from mozi_ai_sdk.FKFD.dataProcess import processWtaData
from mozi_ai_sdk.FKFD.GA_blue import WTA_GA
from mozi_ai_sdk.FKFD.model_red import PN_WTA_red,reward_line
from mozi_ai_sdk.FKFD.model_blue import PN_WTA_blue

parser = argparse.ArgumentParser()
parser.add_argument("--avail_ip_port", type=str, default='127.0.0.1:6060')
parser.add_argument("--platform_mode", type=str, default='eval')
parser.add_argument("--side_name", type=str, default='蓝方')
parser.add_argument("--agent_key_event_file", type=str, default=None)

#  设置墨子安装目录下bin目录为MOZIPATH，程序会自动启动墨子
os.environ['MOZIPATH'] = 'D:\\Mozi\\\MoziServer\\bin'
print(os.environ['MOZIPATH'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
# 拼接路径，指向当前文件夹下的 "model" 文件夹
WTAModelPath = os.path.join(current_dir, "model")
device = torch.device('cpu')

def run(env, blue_step, red_step):
    # 启动墨子服务器，连接墨子服务器，获取初始态势数据
    env.start()
    # 加载想定，初始化推演方
    env.reset()
    # 获取更新态势
    scenario = env.step()
    # 获取推演方，获取本方的所有数据
    # (返回的是本方的所有数据：对应API中的CSide)
    blue_side = scenario.get_side_by_name('蓝方')
    # (返回目标的字典{guid:obe.....})，返回蓝方探测到的目标
    contacts_dic_blue = blue_side.get_contacts()
    # 将蓝方探测到的目标都设为敌对
    temp = [i for i in contacts_dic_blue.values()]
    for i in temp:
        i.set_mark_contact('H')
    # 将推演方准静态化
    blue_side.static_construct()

    red_side = scenario.get_side_by_name('红方')
    contacts_dic_red = red_side.get_contacts()
    temp = [i for i in contacts_dic_red.values()]
    for i in temp:
        i.set_mark_contact('H')
    red_side.static_construct()

    facilities = red_side.get_facilities()
    # 本方的武器初始化可知
    # 基地1的近程武器平台
    B1 = [facility for facility in facilities.values() if '基地1' in facility.strName][0]
    S1 = [facility for facility in facilities.values() if 'S1-1' in facility.strName][0]
    S2 = [facility for facility in facilities.values() if 'S1-2' in facility.strName][0]
    S3 = [facility for facility in facilities.values() if 'S1-3' in facility.strName][0]
    S4 = [facility for facility in facilities.values() if 'S1-4' in facility.strName][0]
    # 基地1的中远程武器平台
    M1 = [facility for facility in facilities.values() if 'M1-1' in facility.strName][0]
    M2 = [facility for facility in facilities.values() if 'M1-2' in facility.strName][0]
    M3 = [facility for facility in facilities.values() if 'M1-3' in facility.strName][0]
    # 基地1的远程武器平台
    L1 = [facility for facility in facilities.values() if 'L1' in facility.strName][0]
    # 基地2的近程武器平台
    B2 = [facility for facility in facilities.values() if '基地2' in facility.strName][0]
    S11 = [facility for facility in facilities.values() if 'S2-1' in facility.strName][0]
    S22 = [facility for facility in facilities.values() if 'S2-2' in facility.strName][0]
    S33 = [facility for facility in facilities.values() if 'S2-3' in facility.strName][0]
    S44 = [facility for facility in facilities.values() if 'S2-4' in facility.strName][0]
    # 基地2的中远程武器平台
    M11 = [facility for facility in facilities.values() if 'M2-1' in facility.strName][0]
    M22 = [facility for facility in facilities.values() if 'M2-2' in facility.strName][0]
    M33 = [facility for facility in facilities.values() if 'M2-3' in facility.strName][0]
    # 基地2的远程武器平台
    L2 = [facility for facility in facilities.values() if 'L2' in facility.strName][0]
    # 基地3的近程武器平台
    B3 = [facility for facility in facilities.values() if '基地3' in facility.strName][0]
    S111 = [facility for facility in facilities.values() if 'S3-1' in facility.strName][0]
    S222 = [facility for facility in facilities.values() if 'S3-2' in facility.strName][0]
    S333 = [facility for facility in facilities.values() if 'S3-3' in facility.strName][0]
    S444 = [facility for facility in facilities.values() if 'S3-4' in facility.strName][0]
    # 基地3的中远程武器平台
    M111 = [facility for facility in facilities.values() if 'M3-1' in facility.strName][0]
    M222 = [facility for facility in facilities.values() if 'M3-2' in facility.strName][0]
    L3 = [facility for facility in facilities.values() if 'L3' in facility.strName][0]
    SL = [facility for facility in facilities.values() if 'S-400E' in facility.strName][0]
    S300_1 = [facility for facility in facilities.values() if 'S300-1' in facility.strName][0]
    S300_2 = [facility for facility in facilities.values() if 'S300-2' in facility.strName][0]
    # 挂架；武器编号；射程；目标高度；目标速度；名称；导弹动力系数；基础命中率；挂架上导弹数；初始总数
    str_obe = {'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'M1': M1, 'M2': M2, 'M3': M3, 'L1': L1,
               'S11': S11, 'S22': S22, 'S33': S33, 'S44': S44, 'M11': M11, 'M22': M22, 'M33': M33, 'L2': L2,
               'S111': S111, 'S222': S222, 'S333': S333, 'S444': S444, 'M111': M111, 'M222': M222, 'L3': L3,
               'SL': SL, 'S300_1': S300_1, 'S300_2': S300_2}
    # 机动性系数（a）：F-16DJ 为4.9，代表非常机动；而导弹类如 炸弹, 高超声速导弹为0，代表不可机动；
    target_a = {'干扰机': 1, '预警机': 1, '女武神无人机': 4, '诡骗丽影无人战斗机': 4, 'F-16DJ战斗机': 4.9, 'B-1B轰炸机': 1, 'B-52H轰炸机': 1.5,
                'RQ-180': 1,
                'F-15E战斗轰炸机': 4.5, '超级眼镜蛇直升机': 2, 'F-16CM战斗机': 4.9, '枪骑兵轰炸机': 2, '巡飞弹': 1, '高超声速导弹': 0, '导弹': 0,
                '炸弹': 0, '侦察机': 1}
    target_name = ['干扰机', '预警机', '女武神无人机', '诡骗丽影无人战斗机', 'F-16DJ战斗机', 'B-1B轰炸机', 'B-52H轰炸机', 'RQ-180', 'F-15E战斗轰炸机',
                   '超级眼镜蛇直升机',
                   'F-16CM战斗机', '枪骑兵轰炸机', '巡飞弹', '高超声速导弹', '导弹', '炸弹', '侦察机']
    special_target = ['高超声速导弹']
    special_target_L = ['超级眼镜蛇直升机', 'F-16DJ', '高超声速导弹']
    global S1_weapon_obe, S1_max_r, S1_weapon_guid, S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
        S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_sum, S1_weapon_every, S1_init_num, \
        S1_target_v, S1_target_h, S1_target_a, Threat_S1, qjk_s1, S1_target_sum, S2_target_v, S2_target_h, S2_target_a, \
        S2_target_sum, Threat_S2, qjk, qjk_s2, S3_target_v, S3_target_h, S3_target_a, S3_target_sum, Threat_S3, qjk_s3, \
        M1_target_v, M1_target_h, M1_target_a, M1_target_sum, Threat_M1, qjk_m1, M2_target_v, M2_target_h, M2_target_a, \
        M2_target_sum, Threat_M2, M3_target_v, M3_target_h, M3_target_a, M3_target_sum, Threat_M3, qjk_m3, L1_target_v, \
        L1_target_h, L1_target_a, L1_target_sum, Threat_L1, qjk_l1, qjk_m2
    global S2_weapon_obe, S2_max_r, S2_weapon_guid, S2_number, S2_weapon_id, S2_weapon_name, S2_weapon_range, S2_weapon_h, \
        S2_weapon_v, S2_rocket, S2_max_v, S2_max_l, S2_pof, S2_weapon_sum, S2_weapon_every, S2_init_num
    global S3_weapon_obe, S3_max_r, S3_weapon_guid, S3_number, S3_weapon_id, S3_weapon_name, S3_weapon_range, S3_weapon_h, \
        S3_weapon_v, S3_rocket, S3_max_v, S3_max_l, S3_pof, S3_weapon_sum, S3_weapon_every, S3_init_num
    global M1_weapon_obe, M1_max_r, M1_weapon_guid, M1_number, M1_weapon_id, M1_weapon_name, M1_weapon_range, M1_weapon_h, \
        M1_weapon_v, M1_rocket, M1_max_v, M1_max_l, M1_pof, M1_weapon_sum, M1_weapon_every, M1_init_num
    global M2_weapon_obe, M2_max_r, M2_weapon_guid, M2_number, M2_weapon_id, M2_weapon_name, M2_weapon_range, M2_weapon_h, \
        M2_weapon_v, M2_rocket, M2_max_v, M2_max_l, M2_pof, M2_weapon_sum, M2_weapon_every, M2_init_num
    global M3_weapon_obe, M3_max_r, M3_weapon_guid, M3_number, M3_weapon_id, M3_weapon_name, M3_weapon_range, M3_weapon_h, \
        M3_weapon_v, M3_rocket, M3_max_v, M3_max_l, M3_pof, M3_weapon_sum, M3_weapon_every, M3_init_num
    global L1_weapon_obe, L1_max_r, L1_weapon_guid, L1_number, L1_weapon_id, L1_weapon_name, L1_weapon_range, L1_weapon_h, \
        L1_weapon_v, L1_rocket, L1_max_v, L1_max_l, L1_pof, L1_weapon_sum, L1_weapon_every, L1_init_num
    global L2_weapon_obe, L2_max_r, L2_weapon_guid, L2_number, L2_weapon_id, L2_weapon_name, L2_weapon_range, L2_weapon_h, \
        L2_weapon_v, L2_rocket, L2_max_v, L2_max_l, L2_pof, L2_weapon_sum, L2_weapon_every, L2_init_num
    global T_weapon_obe, T_max_r, T_weapon_guid, T_number, T_weapon_id, T_weapon_name, T_weapon_range, T_weapon_h, \
        T_weapon_v, T_rocket, T_max_v, T_max_l, T_pof, T_weapon_sum, T_weapon_every, T_init_num
    # (战场环境更新一下)
    env.step()
    # 记录上一次的分配方案
    number = 0
    fired = []
    S1_target_guid, S1_best_plan, S1_no_storage = [], [], []
    S2_target_guid, S2_best_plan, S2_no_storage = [], [], []
    S3_target_guid, S3_best_plan, S3_no_storage = [], [], []
    M1_target_guid, M1_best_plan, M1_no_storage = [], [], []
    M2_target_guid, M2_best_plan, M2_no_storage = [], [], []
    M3_target_guid, M3_best_plan, M3_no_storage = [], [], []
    L1_target_guid, L1_best_plan, L1_no_storage = [], [], []
    L2_target_guid, L2_best_plan, L2_no_storage = [], [], []
    count = 0
    time_count = 190
    NET = 1
    S1_weapon, S2_weapon, S3_weapon = [], [], []
    M1_weapon, M2_weapon, M3_weapon = [], [], []
    L1_weapon, L2_weapon, T_weapon = [], [], []
    S1_weapon_guid, S2_weapon_guid, S3_weapon_guid, M1_weapon_guid = [], [], [], []
    M2_weapon_guid, M3_weapon_guid, L1_weapon_guid, L2_weapon_guid = [], [], [], []
    # 近程范围固有武器
    S1_original_weapon = ['S1', 'S2', 'S3', 'S4']
    S2_original_weapon = ['S11', 'S22', 'S33', 'S44']
    S3_original_weapon = ['S111', 'S222', 'S333', 'S444']
    # 各区域候选武器平台
    S1_prepare = ['S300_1', 'L1', 'M1', 'SL']
    S2_prepare = ['S300_1', 'S300_2', 'M1', 'M111', 'L1', 'L2', 'L3', 'SL']
    S3_prepare = ['S300_2', 'L3', 'M111', 'SL']
    M1_prepare = ['S300_1', 'L1', 'L2', 'L3', 'M11', 'SL', 'M3']
    M2_prepare = ['M1', 'M111', 'M2', 'M222', 'S300_1', 'S300_2', 'L1', 'L2', 'L3', 'SL']
    M3_prepare = ['S300_2', 'L1', 'L2', 'L3', 'M22', 'M33', 'SL']
    L1_prepare = ['SL']
    L2_prepare = ['SL']
    All_weapon = ['L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M11', 'M22', 'M33', 'M111', 'M222', 'S1', 'S2',
                  'S3', 'S4', 'S11', 'S22', 'S33', 'S44', 'S111', 'S222', 'S333', 'S444', 'S300_1', 'S300_2', 'SL']
    All_weapon_obe = [L1, L2, L3, M1, M2, M3, M11, M22, M33, M111, M222, S1, S2,
                      S3, S4, S11, S22, S33, S44, S111, S222, S333, S444, S300_1, S300_2, SL]
    All_weapon_guid = [obe.strGuid for obe in All_weapon_obe]
    every_weapon_mounts = {}
    # 得到初始数量，自动杀伤网时可用
    for w, guid in enumerate(All_weapon_guid):
        current = get_current_num(All_weapon_obe[w].get_weapon_infos())
        every_weapon_mounts[All_weapon[w]] = current

    Base_obe = [B1, B2, B3]
    Base_guid = [i.strGuid for i in Base_obe]
    Des_base = [0, 0, 0]
    # 基地或建筑的价值
    V_a = [[80, 85, 90]]
    # 消除的目标数
    count_hit = 0
    r1 = 0
    r2 = 0

    def update_weapon_S1():
        global S1_weapon_obe, S1_max_r, S1_weapon_guid, S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
            S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_sum, S1_weapon_every, S1_init_num
        if S1_weapon:
            S1_weapon_obe = [str_obe[i] for i in S1_weapon]
            S1_weapon_guid = [obe.strGuid for obe in S1_weapon_obe]
            S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
            S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_every, S1_init_num, S1_max_r = get_weapon_set(
                S1_weapon, weapon_info)
            S1_weapon_sum = sum(S1_number)
        else:
            S1_weapon_obe = []
            S1_weapon_guid = []

    def update_weapon_S2():
        global S2_weapon_obe, S2_max_r, S2_weapon_guid, S2_number, S2_weapon_id, S2_weapon_name, S2_weapon_range, S2_weapon_h, \
            S2_weapon_v, S2_rocket, S2_max_v, S2_max_l, S2_pof, S2_weapon_sum, S2_weapon_every, S2_init_num
        if S2_weapon:
            S2_weapon_obe = [str_obe[i] for i in S2_weapon]
            S2_weapon_guid = [obe.strGuid for obe in S2_weapon_obe]
            S2_number, S2_weapon_id, S2_weapon_name, S2_weapon_range, S2_weapon_h, \
            S2_weapon_v, S2_rocket, S2_max_v, S2_max_l, S2_pof, S2_weapon_every, S2_init_num, S2_max_r = get_weapon_set(
                S2_weapon, weapon_info)
            S2_weapon_sum = sum(S2_number)
        else:
            S2_weapon_obe = []
            S2_weapon_guid = []

    def update_weapon_S3():
        global S3_weapon_obe, S3_max_r, S3_weapon_guid, S3_number, S3_weapon_id, S3_weapon_name, S3_weapon_range, S3_weapon_h, \
            S3_weapon_v, S3_rocket, S3_max_v, S3_max_l, S3_pof, S3_weapon_sum, S3_weapon_every, S3_init_num
        if S3_weapon:
            S3_weapon_obe = [str_obe[i] for i in S3_weapon]
            S3_weapon_guid = [obe.strGuid for obe in S3_weapon_obe]
            S3_number, S3_weapon_id, S3_weapon_name, S3_weapon_range, S3_weapon_h, \
            S3_weapon_v, S3_rocket, S3_max_v, S3_max_l, S3_pof, S3_weapon_every, S3_init_num, S3_max_r = get_weapon_set(
                S3_weapon, weapon_info)
            S3_weapon_sum = sum(S3_number)
        else:
            S3_weapon_obe = []
            S3_weapon_guid = []

    def update_weapon_M1():
        global M1_weapon_obe, M1_max_r, M1_weapon_guid, M1_number, M1_weapon_id, M1_weapon_name, M1_weapon_range, M1_weapon_h, \
            M1_weapon_v, M1_rocket, M1_max_v, M1_max_l, M1_pof, M1_weapon_sum, M1_weapon_every, M1_init_num
        if M1_weapon:
            M1_weapon_obe = [str_obe[i] for i in M1_weapon]
            M1_weapon_guid = [obe.strGuid for obe in M1_weapon_obe]
            M1_number, M1_weapon_id, M1_weapon_name, M1_weapon_range, M1_weapon_h, \
            M1_weapon_v, M1_rocket, M1_max_v, M1_max_l, M1_pof, M1_weapon_every, M1_init_num, M1_max_r = get_weapon_set(
                M1_weapon, weapon_info)
            M1_weapon_sum = sum(M1_number)
        else:
            M1_weapon_obe = []
            M1_weapon_guid = []

    def update_weapon_M2():
        global M2_weapon_obe, M2_max_r, M2_weapon_guid, M2_number, M2_weapon_id, M2_weapon_name, M2_weapon_range, M2_weapon_h, \
            M2_weapon_v, M2_rocket, M2_max_v, M2_max_l, M2_pof, M2_weapon_sum, M2_weapon_every, M2_init_num
        if M2_weapon:
            M2_weapon_obe = [str_obe[i] for i in M2_weapon]
            M2_weapon_guid = [obe.strGuid for obe in M2_weapon_obe]
            M2_number, M2_weapon_id, M2_weapon_name, M2_weapon_range, M2_weapon_h, \
            M2_weapon_v, M2_rocket, M2_max_v, M2_max_l, M2_pof, M2_weapon_every, M2_init_num, M2_max_r = get_weapon_set(
                M2_weapon, weapon_info)
            M2_weapon_sum = sum(M2_number)
        else:
            M2_weapon_obe = []
            M2_weapon_guid = []

    def update_weapon_M3():
        global M3_weapon_obe, M3_max_r, M3_weapon_guid, M3_number, M3_weapon_id, M3_weapon_name, M3_weapon_range, M3_weapon_h, \
            M3_weapon_v, M3_rocket, M3_max_v, M3_max_l, M3_pof, M3_weapon_sum, M3_weapon_every, M3_init_num
        if M3_weapon:
            M3_weapon_obe = [str_obe[i] for i in M3_weapon]
            M3_weapon_guid = [obe.strGuid for obe in M3_weapon_obe]
            M3_number, M3_weapon_id, M3_weapon_name, M3_weapon_range, M3_weapon_h, \
            M3_weapon_v, M3_rocket, M3_max_v, M3_max_l, M3_pof, M3_weapon_every, M3_init_num, M3_max_r = get_weapon_set(
                M3_weapon, weapon_info)
            M3_weapon_sum = sum(M3_number)
        else:
            M3_weapon_obe = []
            M3_weapon_guid = []

    def update_weapon_L1():
        global L1_weapon_obe, L1_max_r, L1_weapon_guid, L1_number, L1_weapon_id, L1_weapon_name, L1_weapon_range, L1_weapon_h, \
            L1_weapon_v, L1_rocket, L1_max_v, L1_max_l, L1_pof, L1_weapon_sum, L1_weapon_every, L1_init_num
        if L1_weapon:
            L1_weapon_obe = [str_obe[i] for i in L1_weapon]
            L1_weapon_guid = [obe.strGuid for obe in L1_weapon_obe]
            L1_number, L1_weapon_id, L1_weapon_name, L1_weapon_range, L1_weapon_h, \
            L1_weapon_v, L1_rocket, L1_max_v, L1_max_l, L1_pof, L1_weapon_every, L1_init_num, L1_max_r = get_weapon_set(
                L1_weapon, weapon_info)
            L1_weapon_sum = sum(L1_number)
        else:
            L1_weapon_obe = []
            L1_weapon_guid = []

    def update_weapon_L2():
        global L2_weapon_obe, L2_max_r, L2_weapon_guid, L2_number, L2_weapon_id, L2_weapon_name, L2_weapon_range, L2_weapon_h, \
            L2_weapon_v, L2_rocket, L2_max_v, L2_max_l, L2_pof, L2_weapon_sum, L2_weapon_every, L2_init_num
        if L2_weapon:
            L2_weapon_obe = [str_obe[i] for i in L2_weapon]
            L2_weapon_guid = [obe.strGuid for obe in L2_weapon_obe]
            L2_number, L2_weapon_id, L2_weapon_name, L2_weapon_range, L2_weapon_h, \
            L2_weapon_v, L2_rocket, L2_max_v, L2_max_l, L2_pof, L2_weapon_every, L2_init_num, L2_max_r = get_weapon_set(
                L2_weapon, weapon_info)
            L2_weapon_sum = sum(L2_number)
        else:
            L2_weapon_obe = []
            L2_weapon_guid = []

    # 暂存发射的武器和目标
    Result_plan = []
    # 每一轮的打击计划
    sub_plan = []
    # 暂不可用武器平台
    Temp_unused = []
    # 总的目标数量
    TARGET = []
    # 武器平台弹药数量
    new_mounts = every_weapon_mounts.copy()
    # 目标轨迹信息
    Target_inf = {}
    last_target_guid = []
    # 统计各区域轮数
    S1_count, S2_count, S3_count = 1, 1, 1
    M1_count, M2_count, M3_count = 1, 1, 1
    L1_count, L2_count, T_count = 1, 1, 1
    # 打印信息用
    S1_time_list, S2_time_list, S3_time_list = [], [], []
    M1_time_list, M2_time_list, M3_time_list = [], [], []
    L1_time_list, L2_time_list, T_time_list = [], [], []
    # 记录打掉的目标
    target_hit = []
    target_hit_name = []

    # 用于分配的武器名称
    keywords = ['空地战术导弹', '直接攻击炸弹', '反坦克导弹', '小直径炸弹']
    flag = False
    step_count = 0

    all_attack_records = []  # 存放每一轮的 attack_records 列表
    all_attack_logs = []  # 存放每一轮的 attack_log 列表
    all_damage_logs = []  # # 存放每一轮的 damage_log 列表
    data_train_blue = []    # 存放每一次推演的数据，包括方案，适应度，logp，critic_est
    algorithm_red = PN_WTA_red(step=red_step)
    red_facilities_in = red_side.get_facilities().values()
    damage_list_0 = get_red_damage(red_facilities_in)
    algorithm_blue = PN_WTA_blue(step=blue_step)

    while True:
        # scenario.set_cur_side_and_dir_view("蓝方", "false")
        step_count += 1  # 更新一步:每一步经过时长有推演倍速决定
        logging.info(f'step_count:{step_count}')
        blue_side.static_update()
        red_side.static_update()

        def blue_move(blue_side, red_side):
            facilities = red_side.get_facilities()
            facilities_info = [
                [facility, facility.strGuid, facility.strName, facility.dLatitude, facility.dLongitude,
                 facility.strDamageState]
                for facility in facilities.values()
            ]
            # logging.info(f'facilities_info:{((facilities_info))}')

            facilities_in_info = []
            facilities_in = []
            for info in facilities_info:
                latitude = float(info[3])
                longitude = float(info[4])
                if 36.75 <= latitude <= 38.7 and 117.1 <= longitude <= 119.6:
                    facilities_in_info.append(info)  # Guid 在索引位置 1
                    facilities_in.append(info[0])
            # logging.info(f'facilities_in_info:{((facilities_in_info))}')

            contacts_dic_blue = blue_side.get_contacts()
            # 将目标信息转为列表结构，每个元素包含：[目标类对象, GUID, 名称, 纬度, 经度]
            targets_info = [
                [target, target.strGuid, target.strName, target.dLatitude, target.dLongitude]
                for target in contacts_dic_blue.values()
                if "防空导弹" not in target.strName
            ]
            # logging.info(f'targets_info:{targets_info}')

            # 实时筛选：区域内目标 Guid 列表
            targets_in_info = []
            target_in = []
            for info in targets_info:
                latitude = float(info[3])
                longitude = float(info[4])
                if 36.75 <= latitude <= 38.7 and 117.1 <= longitude <= 119.6:
                    targets_in_info.append(info)  # Guid 在索引位置 1
                    target_in.append(info[0])
            # logging.info(f'targets_in_info:{(len(targets_in_info))}')

            # 获取有对地攻击能力的飞机类
            acs = blue_side.get_aircrafts()
            acs_assign = [
                ac for ac in acs.values()
                if any(t in ac.strName for t in ['F-16DJ战斗机', 'B-1B轰炸机', '女武神无人机', '枪骑兵轰炸机', '超级眼镜蛇直升机'])
            ]

            # 将飞机信息转为列表结构，每个元素为：[飞机类对象, GUID, 名称, 纬度, 经度]
            acs_assign_info = [
                [ac, ac.strGuid, ac.strName, ac.dLatitude, ac.dLongitude]
                for ac in acs_assign
            ]

            # 筛选武器信息，每个元素为：[ac类, ac GUID, ac 名称, 武器名, 数量, 武器DBID]
            acs_assign_weapon = []
            weapon_num = []
            for ac_info in acs_assign_info:
                ac, guid, name = ac_info[0], ac_info[1], ac_info[2]
                weapon_infos = ac.get_weapon_infos()
                for weapon_str, wid in weapon_infos:
                    if any(k in weapon_str for k in keywords):
                        match = re.match(r'(\d+)x\s+(.*)', weapon_str)
                        if match:
                            count = int(match.group(1))
                            weapon_name = match.group(2)
                            if count > 0:
                                acs_assign_weapon.append([ac, guid, name, weapon_name, count, wid])
                                weapon_num.append(int(count))
                            else:
                                # ac.return_to_base()
                                logging.info(f'飞机{ac.strName}没有武器')
            # logging.info(f'acs_assign_weapon:{acs_assign_weapon}')
            # logging.info(f'weapon_num:{weapon_num}')
            # 无武器返航逻辑
            for item in acs_assign_weapon:
                if item[4] == 0:  # count 在索引 4
                    item[0].return_to_base()  # ac 对象
                    # logging.info(f'飞机{item[2]}没有武器{item[3]}返回基地')

            # 攻击逻辑：生成配对并打击，只执行一次
            # 提前分配，然后是导弹数都变化后再分配
            # 15倍速，完成第一轮打击
            trigger = 220
            data_train_solo = None  # 默认没有数据
            # 如何对结果进行结算？定时实现
            '''
            先结算要比生成前一轮  
            '''
            if step_count == 1 or step_count > trigger and (step_count - trigger) % 30 == 0:
                #  配对产生：设计算法和模型
                if len(weapon_num) > 0 and len(targets_in_info) > 0:
                    pij = pij_generate(targets_in_info, acs_assign_weapon)
                    value = evaluate_targets(targets_in_info, facilities_in_info)
                    # 具体使用多少数量的武器： tij决定
                    # tij = np.full((len(acs_assign_weapon), len(targets_in_info)), 2)
                    tij = build_tij_matrix(acs_assign_weapon, targets_in_info)
                    # logging.info(f'tij长度：{tij}')
                    # logging.info(f'价值评估{value}')
                    # 初始化算法
                    # algorithm_blue = PN_WTA_blue(pij, value, weapon_num, tij, train_step)
                    # pair_mat, pair_plan, data_train_solo = algorithm_blue.run()

                    pair_mat, result, data_train_solo = algorithm_blue.run(pij, value, weapon_num, tij)
                    # 保存网络输出
                    plan = apply_assignment_with_limits(pair_mat, tij, weapon_num)
                    # logging.info(f'产生plan{plan}, 对应适应度{b_fitness}')

                    # 数据库构建
                    # 蓝方对红方的
                    # if step_count != 1:
                    # v1 蓝方对红方的
                    # logging.info(f'facilities_in目标长度：{len(facilities_in)}')
                    # logging.info(f'facilities_in_info目标长度1：{len(facilities_in_info)}')
                    # data_blue = extract_targets_attributes(facilities_in)
                    # logging.info(f'红方数据长度：{len(data_blue[0])}')

                    # v2 蓝方对红方的
                    # data_blue = extract_target_encoded_attributes(facilities_in)

                    # 红方对蓝方的
                    # data = []
                    # for acs in acs_assign_weapon:
                    #     ac = acs[0]
                    #     data_ac = processWtaData(ac)
                    #     data.append(data_ac)
                    # data_red = transpose(data)
                    # logging.info(f'蓝方数据长度：{len(data_red[0])}')
                    # logging.info(f'数据库蓝方：{data_blue}')
                    # logging.info(f'数据库红方：{data_red}')

                    # data_log = data_blue + data_red
                    # data_log.append(len(acs_assign_weapon))
                    # data_log.append(len(targets_in_info))
                    # logging.info(f'acs_assign_weapon蓝方数：{len(acs_assign_weapon)}')
                    # logging.info(f'targets_in_info红方数：{len(targets_in_info)}')
                    # data_log.append(value)
                    # logging.info(f'value长度：{len(value)}')
                    # data_log.append(pij)
                    # data_log.append(tij)
                    # data_log.append(weapon_num)
                    # data_log.append(pair_plan.tolist())
                    # data_log.append(plan.tolist())

                    # logging.info(f'plan行数：{len(plan)}')
                    # logging.info(f'plan列数：{len(plan[0])}\n')

                    # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
                    # custom_headers = ['设施类型名称', '纬度', '经度', '作战范围','射击频率','武器部能力',
                    #                   '设施类型', '武器类型', '装甲类型', '任务类型', '可视类型', '气象条件类型',
                    #                   '损伤情况', '重要性', '急迫性',
                    #                   '空中单位名称', '目标类型', '经纬度', '速度', '方位角', '作战范围',
                    #                   '武器数量', '目标数量', '威胁值', '打击概率', '损伤概率', '可行性', '分配方案']  # 自定义表头
                    # df = pd.DataFrame([data_log], columns=custom_headers)
                    # file_path = '蓝方数据库输出v2.xlsx'

                    # if len(data_blue[0]) == len(targets_in_info):
                    #     # Excel 文件路径
                    #     file_path = '蓝方数据库.xlsx'
                    #     sheet_name = 'Sheet1'
                    #     df = pd.DataFrame([data_log])  # 假设 data_log 是一个 dict
                    #
                    #     # 判断文件是否存在
                    #     if os.path.exists(file_path):
                    #         # 打开已有 Excel 文件
                    #         book = load_workbook(file_path)
                    #
                    #         # 启动 ExcelWriter 并追加
                    #         with pd.ExcelWriter(file_path, engine='openpyxl', mode='a',
                    #                             if_sheet_exists='overlay') as writer:
                    #             writer.book = book
                    #             writer.sheets = {ws.title: ws for ws in book.worksheets}
                    #
                    #             # 获取目标 Sheet 当前最大行号（从0开始）
                    #             if sheet_name in writer.sheets:
                    #                 start_row = writer.sheets[sheet_name].max_row
                    #             else:
                    #                 start_row = 0
                    #
                    #             #  只有 start_row=0 时写 header，其他情况只写数据
                    #             df.to_excel(writer,
                    #                         sheet_name=sheet_name,
                    #                         startrow=start_row,
                    #                         index=False,
                    #                         header=(start_row == 0))  # 只第一次写表头
                    #     else:
                    #         # 第一次写文件
                    #         df.to_excel(file_path, sheet_name=sheet_name, index=False)
                    # 蓝方数据库构建结束

                    # 依据打击方案，记录分配情况
                    attack_records = []
                    for i in range(len(plan)):
                        row = plan[i]

                        ac, guid, name, weapon_name, count, wid = acs_assign_weapon[i]

                        for j, num in enumerate(row):
                            if num > 0 and j < len(targets_in_info):
                                target_obj, target_guid, target_name, target_lat, target_lon = targets_in_info[j]

                                ac.manual_attack(target_guid, wid, num)
                                # logging.info(f"飞机 {name}（编号: {i}） 使用武器 {weapon_name}（数量: {num}） 攻击目标 {target_name}（编号: {j}）")
                                attack_records.append([
                                    ac, guid, name,
                                    target_obj, target_guid, target_name, target_lat, target_lon,
                                    weapon_name, wid, num
                                ])
                                break  # 只处理一个非零元素

                    all_attack_records.append(attack_records)

            # 结果结算和监控
            # if step_count == 1 or (step_count > 300 and step_count % 101 == 0):
            if step_count > trigger and (step_count - trigger + 1) % 30 == 0:
                # 毁伤情况记录
                attack_log = monitor_attack_results(all_attack_records[-1], facilities_in_info)
                # 损伤情况记录
                damage_log = monitor_aircraft_damage(all_attack_records[-1], acs_assign_info)
                # 保存到总集合中
                all_attack_logs.append(attack_log)
                all_damage_logs.append(damage_log)
                # logging.info(f'attack_logs:{all_attack_logs}')
                # logging.info(f'damage_logs:{all_damage_logs}')

            # 根据发射的武器确定取消积压命令和避免浪费
            # 没分配成功的要手动取消
            # unit_drop_target_contact(t)

            # if step_count == 450:
            #     logging.info(f'all_attack_records:{all_attack_records}')
            #     logging.info(f'all_attack_logs:{all_attack_logs}')
            #     logging.info(f'all_damage_logs:{all_damage_logs}')

            return data_train_solo

        data_train_solo = blue_move(blue_side, red_side)
        # 只有真正产生时才收集
        if data_train_solo is not None:
            data_train_blue.append(data_train_solo)
            # logging.info(f'data_train_solo:{data_train_solo}')
        count = count + 1
        # 基地价值阶段
        for k, v in enumerate(Base_guid):
            if red_side.get_unit_by_guid(v) is None:
                Des_base[k] = 1 * V_a[0][k]
            else:
                Des_base[k] = 0.01 * float(Base_obe[k].strDamageState) * V_a[0][k]

        Des = sum(Des_base) / sum(V_a[0])
        contacts_dic = red_side.contacts
        # 目标速度大于5判定为空中目标
        targets = [item for item in contacts_dic.values() if item.fCurrentSpeed > 5]
        for i in targets:
            i.set_mark_contact('H')
        targets_copy = targets.copy()
        targets_guid = [T.strGuid for T in targets]
        TARGET = TARGET + targets_guid
        TARGET = list(set(TARGET))
        # 得到目标近期的信息，存储在字典嵌套列表中
        if targets_guid:
            for t in targets_guid:
                if t not in last_target_guid:
                    Target_inf[t] = []
            for target in targets:
                temp = Target_inf[target.strGuid][-19:]
                temp.append([target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                             target.dLatitude, target.fCurrentHeading, scenario.get_current_time()])
                Target_inf[target.strGuid] = temp
            # 消失的目标信息消失
            for guid in TARGET:
                if not scenario.unit_is_alive(guid):
                    if guid in Target_inf:
                        del Target_inf[guid]
        last_target_guid = targets_guid
        # 已被分配武器的目标不作为下一阶段打击的目标
        for g in fired:
            if g in targets_guid:
                index = targets_guid.index(g)
                temp_t = targets_copy[index]
                if temp_t in targets:
                    targets.remove(temp_t)
        S1_target, S2_target, S3_target, M1_target, M2_target, M3_target, L1_target, L2_target = [], [], [], [], [], [], [], []
        # 进行目标的划分
        for target in targets:
            # 目标纬度
            latitude = float(target.dLatitude)
            # 目标经度
            longitude = float(target.dLongitude)
            if longitude >= 119.6:
                pass
            # 中程#2
            elif 118.07 <= longitude < 118.93 and 37.33 <= latitude < 38.07:
                M2_target.append(target)
            # 远程范围
            elif 117.1 <= longitude < 119.6:
                # 中程#1
                if 117.8 <= longitude < 118.67 and 37.82 <= latitude < 38.3:
                    M1_target.append(target)
                # 中程#3
                elif 117.8 <= longitude < 118.67 and 37.13 <= latitude < 37.58:
                    M3_target.append(target)
                # 近程#2
                elif 117.8 <= longitude < 118.07 and 37.58 <= latitude < 37.82:
                    S2_target.append(target)
                # 近程#1
                elif 117.4 <= longitude < 117.8 and 37.7 <= latitude < 38.15:
                    S1_target.append(target)
                # 近程#1
                elif 117.4 <= longitude < 117.8 and 37.3 <= latitude < 37.7:
                    S3_target.append(target)
                # 远程#1
                elif 37.7 <= latitude < 38.7:
                    L1_target.append(target)
                elif 36.75 <= latitude < 37.7:
                    L2_target.append(target)
        # print("a")
        # 各防空区域目标的guid
        S1_target_guid = [target.strGuid for target in S1_target]
        S2_target_guid = [target.strGuid for target in S2_target]
        S3_target_guid = [target.strGuid for target in S3_target]
        M1_target_guid = [target.strGuid for target in M1_target]
        M2_target_guid = [target.strGuid for target in M2_target]
        M3_target_guid = [target.strGuid for target in M3_target]
        L1_target_guid = [target.strGuid for target in L1_target]
        L2_target_guid = [target.strGuid for target in L2_target]
        # 检测各单元存活,并更新弹药数量
        for w, guid in enumerate(All_weapon_guid):
            if red_side.get_unit_by_guid(guid) is None:
                Temp_unused.append(All_weapon[w])
            else:
                current = get_current_num(All_weapon_obe[w].get_weapon_infos())
                new_mounts[All_weapon[w]] = current
                if current == 0:
                    Temp_unused.append(All_weapon[w])
        # 不可用的武器置0
        for i in Temp_unused:
            new_mounts[i] = 0
        S1_weapon = []
        S2_weapon = []
        S3_weapon = []
        M1_weapon = []
        M2_weapon = []
        M3_weapon = []
        L1_weapon = []
        L2_weapon = []
        Temp_unused = []
        # 记录打击的目标
        last_target_weapon = []

        # 信息融合、威胁评估模块
        S1_target_sum, Threat_S1, qjk_s1, S1_target_v, S1_target_h, S1_target_name, S1_target_a, S1_class_num = None, [], None, None, None, None, None, {}
        if S1_target:
            # 目标数量
            S1_target_sum = len(S1_target)
            # 各目标速度：list
            S1_target_v = [target.fCurrentSpeed for target in S1_target]
            # 各目标高度：list
            S1_target_h = [target.fCurrentAltitude_ASL for target in S1_target]
            # 目标名字
            S1_target_name = [target.strName for target in S1_target]
            # 目标机动系数
            S1_target_a, S1_target_class = get_target_A(S1_target_name, target_name, target_a)
            S1_class_num = get_class_num(S1_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            S1_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in S1_target]
            Threat_S1, Damage_S1, Base_S1 = WNN_TA(S1_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, S1_target_sum])[0]
            qjk_s1 = np.zeros((S1_target_sum, 3))
            for k, v in enumerate(Base_S1):
                qjk_s1[k][v] = Damage_S1[k]
        # 信息融合、威胁评估模块
        S2_target_sum, Threat_S2, qjk_s2, S2_target_v, S2_target_h, S2_target_name, S2_target_a, S2_class_num = None, [], None, None, None, None, None, {}
        if S2_target:
            S2_target_sum = len(S2_target)
            S2_target_v = [target.fCurrentSpeed for target in S2_target]
            S2_target_h = [target.fCurrentAltitude_ASL for target in S2_target]
            S2_target_name = [target.strName for target in S2_target]
            # 目标机动系数
            S2_target_a, S2_target_class = get_target_A(S2_target_name, target_name, target_a)
            S2_class_num = get_class_num(S2_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            S2_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in S2_target]
            Threat_S2, Damage_S2, Base_S2 = WNN_TA(S2_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, S2_target_sum])[0]
            qjk_s2 = np.zeros((S2_target_sum, 3))
            for k, v in enumerate(Base_S2):
                qjk_s2[k][v] = Damage_S2[k]
        # 信息融合、威胁评估模块
        S3_target_sum, Threat_S3, qjk_s3, S3_target_v, S3_target_h, S3_target_name, S3_target_a, S3_class_num = None, [], None, None, None, None, None, {}
        if S3_target:
            S3_target_sum = len(S3_target)
            S3_target_v = [target.fCurrentSpeed for target in S3_target]
            S3_target_h = [target.fCurrentAltitude_ASL for target in S3_target]
            S3_target_name = [target.strName for target in S3_target]
            # 目标机动系数
            S3_target_a, S3_target_class = get_target_A(S3_target_name, target_name, target_a)
            S3_class_num = get_class_num(S3_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            S3_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in S3_target]
            Threat_S3, Damage_S3, Base_S3 = WNN_TA(S3_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, S3_target_sum])[0]
            qjk_s3 = np.zeros((S3_target_sum, 3))
            for k, v in enumerate(Base_S3):
                qjk_s3[k][v] = Damage_S3[k]
        # 信息融合、威胁评估模块
        M1_target_sum, Threat_M1, qjk_m1, M1_target_v, M1_target_h, M1_target_name, M1_target_a, M1_class_num = None, [], None, None, None, None, None, {}
        if M1_target:
            M1_target_sum = len(M1_target)
            M1_target_v = [target.fCurrentSpeed for target in M1_target]
            M1_target_h = [target.fCurrentAltitude_ASL for target in M1_target]
            M1_target_name = [target.strName for target in M1_target]
            # 目标机动系数
            M1_target_a, M1_target_class = get_target_A(M1_target_name, target_name, target_a)
            M1_class_num = get_class_num(M1_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            M1_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in M1_target]
            Threat_M1, Damage_M1, Base_M1 = WNN_TA(M1_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, M1_target_sum])[0]
            qjk_m1 = np.zeros((M1_target_sum, 3))
            for k, v in enumerate(Base_M1):
                qjk_m1[k][v] = Damage_M1[k]
        # 信息融合、威胁评估模块
        M2_target_sum, Threat_M2, qjk_m2, M2_target_v, M2_target_h, M2_target_name, M2_target_a, M2_class_num = None, [], None, None, None, None, None, {}
        if M2_target:
            M2_target_sum = len(M2_target)
            M2_target_v = [target.fCurrentSpeed for target in M2_target]
            M2_target_h = [target.fCurrentAltitude_ASL for target in M2_target]
            M2_target_name = [target.strName for target in M2_target]
            # 目标机动系数
            M2_target_a, M2_target_class = get_target_A(M2_target_name, target_name, target_a)
            M2_class_num = get_class_num(M2_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            M2_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in M2_target]
            Threat_M2, Damage_M2, Base_M2 = WNN_TA(M2_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, M2_target_sum])[0]
            qjk_m2 = np.zeros((M2_target_sum, 3))
            for k, v in enumerate(Base_M2):
                qjk_m2[k][v] = Damage_M2[k]
        # 信息融合、威胁评估模块
        M3_target_sum, Threat_M3, qjk_m3, M3_target_v, M3_target_h, M3_target_name, M3_target_a, M3_class_num = None, [], None, None, None, None, None, {}
        if M3_target:
            M3_target_sum = len(M3_target)
            M3_target_v = [target.fCurrentSpeed for target in M3_target]
            M3_target_h = [target.fCurrentAltitude_ASL for target in M3_target]
            M3_target_name = [target.strName for target in M3_target]
            # 目标机动系数
            M3_target_a, M3_target_class = get_target_A(M3_target_name, target_name, target_a)
            M3_class_num = get_class_num(M3_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            M3_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in M3_target]
            Threat_M3, Damage_M3, Base_M3 = WNN_TA(M3_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, M3_target_sum])[0]
            qjk_m3 = np.zeros((M3_target_sum, 3))
            for k, v in enumerate(Base_M3):
                qjk_m3[k][v] = Damage_M3[k]
        # 信息融合、威胁评估模块
        L1_target_sum, Threat_L1, qjk_l1, L1_target_v, L1_target_h, L1_target_name, L1_target_a, L1_class_num = None, [], None, None, None, None, None, {}
        if L1_target:
            L1_target_sum = len(L1_target)
            L1_target_v = [target.fCurrentSpeed for target in L1_target]
            L1_target_h = [target.fCurrentAltitude_ASL for target in L1_target]
            L1_target_name = [target.strName for target in L1_target]
            # 目标机动系数
            L1_target_a, L1_target_class = get_target_A(L1_target_name, target_name, target_a)
            L1_class_num = get_class_num(L1_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            L1_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in L1_target]
            Threat_L1, Damage_L1, Base_L1 = WNN_TA(L1_target_inf).run()
            # 生成目标威胁值
            # Threat_L1 = [1 for d in range(1, L1_target_sum + 1)]
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, L1_target_sum])[0]
            qjk_l1 = np.zeros((L1_target_sum, 3))
            for k, v in enumerate(Base_L1):
                qjk_l1[k][v] = Damage_L1[k]
        L2_target_sum, Threat_L2, qjk_l2, L2_target_v, L2_target_h, L2_target_name, L2_target_a, L2_class_num = None, [], None, None, None, None, None, {}
        if L2_target:
            L2_target_sum = len(L2_target)
            L2_target_v = [target.fCurrentSpeed for target in L2_target]
            L2_target_h = [target.fCurrentAltitude_ASL for target in L2_target]
            L2_target_name = [target.strName for target in L2_target]
            # 目标机动系数
            L2_target_a, L2_target_class = get_target_A(L2_target_name, target_name, target_a)
            L2_class_num = get_class_num(L2_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[名称, 速度，高度，经度，纬度，朝向]
            L2_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude,
                 target.dLatitude,
                 target.fCurrentHeading] for target in L2_target]
            Threat_L2, Damage_L2, Base_L2 = WNN_TA(L2_target_inf).run()
            # 生成目标威胁值
            # Threat_L2 = [1 for d in range(1, L2_target_sum + 1)]
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, L2_target_sum])[0]
            qjk_l2 = np.zeros((L2_target_sum, 3))
            for k, v in enumerate(Base_L2):
                qjk_l2[k][v] = Damage_L2[k]
            # print("16")
        # 根据各区域的威胁、数量、种类等分配合适的武器
        # 根据特殊目标分配特殊武器,并执行本地任务
        # new_mounts = every_weapon_mounts.copy()
        # print("c")
        if NET == 1:
            # 本地任务
            S1_new = {}
            S1_target_norm = 0
            S1_now_sum = 0
            if S1_target:
                for i in S1_original_weapon:
                    if new_mounts[i] > 0:
                        S1_new[i] = new_mounts[i]
                        S1_now_sum = S1_now_sum + new_mounts[i]
                if new_mounts['SL']:
                    temp = 0
                    for c, n in S1_class_num.items():
                        if c in special_target:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    S1_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    S1_target_norm = max(0, S1_target_sum - temp)
                else:
                    S1_target_norm = S1_target_sum
                if S1_now_sum > S1_target_norm:
                    S1_target_norm = 0
                else:
                    S1_target_norm = S1_target_norm - S1_now_sum
            # 本地任务
            S2_new = {}
            S2_target_norm = 0
            S2_now_sum = 0
            if S2_target:
                # 本地任务
                for i in S2_original_weapon:
                    if new_mounts[i] > 0:
                        S2_new[i] = new_mounts[i]
                        S2_now_sum = S2_now_sum + new_mounts[i]
                if new_mounts['SL']:
                    temp = 0
                    for c, n in S2_class_num.items():
                        if c in special_target:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    S2_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    S2_target_norm = max(0, S2_target_sum - temp)
                else:
                    S2_target_norm = S2_target_sum
                if S2_now_sum > S2_target_norm:
                    S2_target_norm = 0
                else:
                    S2_target_norm = S2_target_norm - S2_now_sum
            # 本地任务
            S3_new = {}
            S3_target_norm = 0
            S3_now_sum = 0
            if S3_target:
                # 本地任务
                for i in S3_original_weapon:
                    if new_mounts[i] > 0:
                        S3_new[i] = new_mounts[i]
                        S3_now_sum = S3_now_sum + new_mounts[i]
                if new_mounts['SL']:
                    temp = 0
                    for c, n in S3_class_num.items():
                        if c in special_target:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    S3_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    S3_target_norm = max(0, S3_target_sum - temp)
                else:
                    S3_target_norm = S3_target_sum
                if S3_now_sum > S3_target_norm:
                    S3_target_norm = 0
                else:
                    S3_target_norm = S3_target_norm - S3_now_sum
            # 本地任务
            M1_new = {}
            M1_target_norm = 0
            if M1_target:
                # 如果有S400且有目标
                if new_mounts['SL']:
                    temp = 0
                    # 计算特殊目标的总数量
                    for c, n in M1_class_num.items():
                        if c in special_target:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    # 武器平台添加最大的特殊数量或者是目标数量
                    M1_new['SL'] = num
                    # 总资源更新
                    new_mounts['SL'] = new_mounts['SL'] - num
                    # 剩余普通目标数量
                    M1_target_norm = M1_target_sum - temp
                else:
                    M1_target_norm = M1_target_sum
                # 执行本地任务
                if M1_target_norm:
                    num_m1 = new_mounts['M1']
                    num_m2 = new_mounts['M2']
                    num_m3 = new_mounts['M3']  # ✅ 加入 M3

                    total_available = num_m1 + num_m2 + num_m3

                    # 情况 1：目标数量 >= 所有资源，总量全部打光
                    if M1_target_norm >= total_available:
                        M1_new['M1'] = num_m1
                        M1_new['M2'] = num_m2
                        M1_new['M3'] = num_m3

                        new_mounts['M1'] = 0
                        new_mounts['M2'] = 0
                        new_mounts['M3'] = 0

                        M1_target_norm -= total_available

                    # 情况 2：目标数量不足，按优先级消耗 M1 → M2 → M3
                    else:
                        # 给每种武器评分（可根据实际能力调整权重）
                        score_M1 = num_m1 * 0.3
                        score_M2 = num_m2 * 0.3
                        score_M3 = num_m3 * 0.3

                        total_score = score_M1 + score_M2 + score_M3

                        if total_score == 0:
                            continue  # 没有可用武器，跳过

                        # 计算比例
                        ratio_M1 = score_M1 / total_score
                        ratio_M2 = score_M2 / total_score
                        ratio_M3 = score_M3 / total_score

                        # 分配任务数
                        assign_M1 = int(M1_target_norm * ratio_M1)
                        assign_M2 = int(M1_target_norm * ratio_M2)
                        assign_M3 = M1_target_norm - assign_M1 - assign_M2  # 保底补足总数

                        # 限制不超过库存
                        assign_M1 = min(assign_M1, num_m1)
                        assign_M2 = min(assign_M2, num_m2)
                        assign_M3 = min(assign_M3, num_m3)

                        # 记录分配
                        M1_new['M1'] = assign_M1
                        M1_new['M2'] = assign_M2
                        M1_new['M3'] = assign_M3

                        # 更新库存
                        new_mounts['M1'] -= assign_M1
                        new_mounts['M2'] -= assign_M2
                        new_mounts['M3'] -= assign_M3

                        M1_target_norm = 0
            # 本地任务
            M2_new = {}
            M2_target_norm = 0
            if M2_target:
                if new_mounts['SL']:
                    temp = 0
                    for c, n in M2_class_num.items():
                        if c in special_target:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    M2_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    M2_target_norm = M2_target_sum - temp
                else:
                    M2_target_norm = M2_target_sum
                # 执行本地任务
                if M2_target_norm:
                    num_m1 = new_mounts['M11']
                    num_m2 = new_mounts['M22']
                    num_m3 = new_mounts['M33']  # 新增M33

                    total_available = num_m1 + num_m2 + num_m3

                    if M2_target_norm >= total_available:
                        M2_new['M11'] = num_m1
                        M2_new['M22'] = num_m2
                        M2_new['M33'] = num_m3

                        new_mounts['M11'] = 0
                        new_mounts['M22'] = 0
                        new_mounts['M33'] = 0

                        M2_target_norm -= total_available
                    else:
                        # 给每种武器评分（你可以根据实际情况调整权重）
                        score_M11 = num_m1 * 0.3
                        score_M22 = num_m2 * 0.3
                        score_M33 = num_m3 * 0.3

                        total_score = score_M11 + score_M22 + score_M33

                        if total_score == 0:
                            continue  # 所有武器都没资源了

                        # 按比例分配目标
                        ratio_M11 = score_M11 / total_score
                        ratio_M22 = score_M22 / total_score
                        ratio_M33 = score_M33 / total_score

                        assign_M11 = int(M2_target_norm * ratio_M11)
                        assign_M22 = int(M2_target_norm * ratio_M22)
                        assign_M33 = M2_target_norm - assign_M11 - assign_M22  # 保证总和不超

                        # 不超过库存限制
                        assign_M11 = min(assign_M11, num_m1)
                        assign_M22 = min(assign_M22, num_m2)
                        assign_M33 = min(assign_M33, num_m3)

                        # 分配任务
                        M2_new['M11'] = assign_M11
                        M2_new['M22'] = assign_M22
                        M2_new['M33'] = assign_M33

                        # 更新库存
                        new_mounts['M11'] -= assign_M11
                        new_mounts['M22'] -= assign_M22
                        new_mounts['M33'] -= assign_M33

                        M2_target_norm = 0
            # 本地任务
            M3_new = {}
            M3_target_norm = 0
            if M3_target:
                if new_mounts['SL']:
                    temp = 0
                    for c, n in M3_class_num.items():
                        if c in special_target:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    M3_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    M3_target_norm = M3_target_sum - temp
                else:
                    M3_target_norm = M3_target_sum
                # 执行本地任务
                if M3_target_norm:
                    num_m1 = new_mounts['M111']
                    num_m2 = new_mounts['M222']
                    # 目标数量过多，后期需要请求调用别处资源
                    if M3_target_norm >= num_m1 + num_m2:
                        M3_new['M111'] = num_m1
                        M3_new['M222'] = num_m2
                        new_mounts['M111'] = 0
                        new_mounts['M222'] = 0
                        M3_target_norm = M3_target_norm - num_m1 - num_m2
                    else:
                        # 给武器评分（可调整权重）
                        score_M111 = num_m1 * 0.5
                        score_M222 = num_m2 * 0.5
                        total_score = score_M111 + score_M222

                        if total_score == 0:
                            continue  # 没资源可分配，跳过

                        # 分配比例
                        ratio_M111 = score_M111 / total_score
                        ratio_M222 = score_M222 / total_score

                        assign_M111 = int(M3_target_norm * ratio_M111)
                        assign_M222 = M3_target_norm - assign_M111

                        # 不超过库存
                        assign_M111 = min(assign_M111, num_m1)
                        assign_M222 = min(assign_M222, num_m2)

                        # 分配结果
                        M3_new['M111'] = assign_M111
                        M3_new['M222'] = assign_M222

                        # 更新库存
                        new_mounts['M111'] -= assign_M111
                        new_mounts['M222'] -= assign_M222

                        M3_target_norm = 0
            # 本地任务
            L1_new = {}
            L1_target_norm = 0
            if L1_target:
                if new_mounts['SL']:
                    temp = 0
                    for c, n in L1_class_num.items():
                        if c in special_target_L:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    L1_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    L1_target_norm = L1_target_sum - temp
                else:
                    L1_target_norm = L1_target_sum
                # 执行本地任务
                if L1_target_norm:
                    num_m1 = new_mounts['L1']
                    num_m2 = new_mounts['L2']
                    # 目标数量过多，后期需要请求调用别处资源
                    if L1_target_norm >= num_m1 + num_m2:
                        L1_new['L1'] = num_m1
                        L1_new['L2'] = num_m2
                        new_mounts['L1'] = 0
                        new_mounts['L2'] = 0
                        L1_target_norm = L1_target_norm - num_m1 - num_m2
                    else:
                        # 定义评分（可根据实际武器能力微调）
                        score_L1 = num_m1 * 0.5
                        score_L2 = num_m2 * 0.5
                        total_score = score_L1 + score_L2

                        if total_score == 0:
                            continue  # 无资源跳过

                        # 计算比例
                        ratio_L1 = score_L1 / total_score
                        ratio_L2 = score_L2 / total_score

                        # 按比例分配目标
                        assign_L1 = int(L1_target_norm * ratio_L1)
                        assign_L2 = L1_target_norm - assign_L1

                        # 不超过实际库存
                        assign_L1 = min(assign_L1, num_m1)
                        assign_L2 = min(assign_L2, num_m2)

                        # 分配任务
                        L1_new['L1'] = assign_L1
                        L1_new['L2'] = assign_L2

                        # 更新库存
                        new_mounts['L1'] -= assign_L1
                        new_mounts['L2'] -= assign_L2

                        # 任务完成
                        L1_target_norm = 0
            # 本地任务
            L2_new = {}
            L2_target_norm = 0
            if L2_target:
                if new_mounts['SL']:
                    temp = 0
                    for c, n in L2_class_num.items():
                        if c in special_target_L:
                            temp = temp + n
                    num = min(temp, new_mounts['SL'])
                    L2_new['SL'] = num
                    new_mounts['SL'] = new_mounts['SL'] - num
                    L2_target_norm = L2_target_sum - temp
                else:
                    L2_target_norm = L2_target_sum
                # 执行本地任务
                if L2_target_norm:
                    num_m1 = new_mounts['L3']
                    num_m2 = new_mounts['L2']
                    # 目标数量过多，后期需要请求调用别处资源
                    if L2_target_norm >= num_m1 + num_m2:
                        L2_new['L3'] = num_m1
                        L2_new['L2'] = num_m2
                        new_mounts['L3'] = 0
                        new_mounts['L2'] = 0
                        L2_target_norm = L2_target_norm - num_m1 - num_m2
                    # 目标数量小，可剩余资源供后期别处使用 —— 改为按评分选择武器系统
                    else:
                        # 为每种武器评分：你可以根据速度、命中概率、弹道性能等自定义
                        # 示例：我们先简单用剩余数量作为评分基础
                        score_L3 = num_m1 * 0.5  # 你可以换成自定义函数
                        score_L2 = num_m2 * 0.5  # 例：略微降低权重

                        total_score = score_L3 + score_L2

                        if total_score == 0:
                            continue  # 没有资源可分配，跳过

                        ratio_L3 = score_L3 / total_score
                        ratio_L2 = score_L2 / total_score

                        assign_L3 = int(L2_target_norm * ratio_L3)
                        assign_L2 = L2_target_norm - assign_L3

                        # 不能超过库存
                        assign_L3 = min(assign_L3, num_m1)
                        assign_L2 = min(assign_L2, num_m2)

                        L2_new['L3'] = assign_L3
                        L2_new['L2'] = assign_L2

                        new_mounts['L3'] -= assign_L3
                        new_mounts['L2'] -= assign_L2

                        L2_target_norm = 0

            # 执行支援任务(近程、中程、远程，内部由威胁度决定顺序)
            A_target_norm = [S1_target_norm, S2_target_norm, S3_target_norm, M1_target_norm, M2_target_norm,
                             M3_target_norm,
                             L1_target_norm, L2_target_norm]
            A_prepare = [S1_prepare, S2_prepare, S3_prepare, M1_prepare, M2_prepare, M3_prepare, L1_prepare,
                         L2_prepare]
            A_new = [S1_new, S2_new, S3_new, M1_new, M2_new, M3_new, L1_new, L2_new]
            S_Threat = [sum(Threat_S1), sum(Threat_S2), sum(Threat_S3)]
            M_Threat = [sum(Threat_M1), sum(Threat_M2), sum(Threat_M3)]
            L_Threat = [sum(Threat_L1), sum(Threat_L2)]
            A_Threat = [S_Threat, M_Threat, L_Threat]
            A_sort = [0, 3, 6]
            A_go = []
            for p, threat in enumerate(A_Threat):
                sorted_id = sorted(range(len(threat)), key=lambda k: threat[k], reverse=True)
                sort = [i + A_sort[p] for i in sorted_id]
                A_go = A_go + sort
            for i in A_go:
                if A_target_norm[i]:
                    for w in A_prepare[i]:
                        t = A_target_norm[i]
                        A_target_norm[i] -= new_mounts[w]
                        if A_target_norm[i] <= 0:
                            if w in A_new[i]:
                                A_new[i][w] = A_new[i][w] + t
                            else:
                                A_new[i][w] = t
                            new_mounts[w] = - A_target_norm[i]
                            break
                        else:
                            if w in A_new[i]:
                                A_new[i][w] = A_new[i][w] + new_mounts[w]
                            else:
                                A_new[i][w] = new_mounts[w]
                            new_mounts[w] = 0
            # 武器平台组合的预处理模块
            for k, v in S1_new.items():
                if v > 0:
                    S1_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_S1()
            for k, v in S2_new.items():
                if v > 0:
                    S2_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_S2()
            for k, v in S3_new.items():
                if v > 0:
                    S3_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_S3()
            for k, v in M1_new.items():
                if v > 0:
                    M1_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_M1()
            for k, v in M2_new.items():
                if v > 0:
                    M2_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_M2()
            for k, v in M3_new.items():
                if v > 0:
                    M3_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_M3()
            for k, v in L1_new.items():
                if v > 0:
                    L1_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_L1()
            for k, v in L2_new.items():
                if v > 0:
                    L2_weapon.append(k)
                    weapon_info[k]['num_weapon'] = v
            update_weapon_L2()
        else:
            S1_new = {}
            S1_weapon_old = ['S1', 'S2', 'S3', 'S4', 'S300_1']
            for i in S1_weapon_old:
                if new_mounts[i] > 0:
                    S1_new[i] = new_mounts[i]
                    S1_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_S1()
            S2_new = {}
            S2_weapon_old = ['S11', 'S22', 'S44', 'S33']
            for i in S2_weapon_old:
                if new_mounts[i] > 0:
                    S2_new[i] = new_mounts[i]
                    S2_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_S2()
            S3_new = {}
            S3_weapon_old = ['S111', 'S222', 'S333', 'S444', 'S300_2']
            for i in S3_weapon_old:
                if new_mounts[i] > 0:
                    S3_new[i] = new_mounts[i]
                    S3_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_S3()
            M1_new = {}
            M1_weapon_old = ['M1', 'M2', 'M3']
            for i in M1_weapon_old:
                if new_mounts[i] > 0:
                    M1_new[i] = new_mounts[i]
                    M1_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_M1()
            M2_new = {}
            M2_weapon_old = ['M11', 'M22', 'M33']
            for i in M2_weapon_old:
                if new_mounts[i] > 0:
                    M2_new[i] = new_mounts[i]
                    M2_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_M2()
            M3_new = {}
            M3_weapon_old = ['M111', 'M222']
            for i in M3_weapon_old:
                if new_mounts[i] > 0:
                    M3_new[i] = new_mounts[i]
                    M3_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_M3()
            L1_new = {}
            L1_weapon_old = ['L1', 'SL']
            for i in L1_weapon_old:
                if new_mounts[i] > 0:
                    L1_new[i] = new_mounts[i]
                    L1_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_L1()
            L2_new = {}
            L2_weapon_old = ['L2', 'L3']
            for i in L2_weapon_old:
                if new_mounts[i] > 0:
                    L2_new[i] = new_mounts[i]
                    L2_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_L2()
        # print("d")
        # 武器目标分配模块
        if (S1_target and S1_weapon) and count % 2 == 0 or '高超声速导弹' in S1_class_num:
            start = timeit.default_timer()
            dis_S = []
            for i, j in enumerate(S1_weapon):
                dis_S = dis_S + [[float(S1_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in S1_target]] * \
                        S1_number[i]
            # 生成可行性矩阵
            Fij_S1 = feasibility(S1_weapon_range, dis_S, S1_weapon_v, S1_target_v, S1_weapon_h, S1_target_h,
                                 S1_target_name, S1_new)
            # 生成打击概率矩阵
            Pij_S1, Fij_SA1 = probability_of_hit(S1_rocket, S1_max_v, S1_max_l, S1_target_v, dis_S, S1_pof,
                                                 S1_target_a)
            Fij_1 = Fij_S1 * Fij_SA1
            S1_new_weapon_obe = [str_obe[key] for key in S1_new.keys()]
            weapon_system_mapping = []
            for idx, (weapon_system, count) in enumerate(S1_new.items()):
                weapon_system_mapping.extend([idx] * count)

            # s_t = []
            # for target_index, target_inf in enumerate(S1_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)

            # # [ 名称， 目标类型， 经度， 纬度， 方位角，打击范围， 威胁值]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(S1_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(S1_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(S1_weapon_sum)
            # data_final.append(S1_target_sum)
            # data_final.append(Threat_S1)
            # data_final.append(Pij_S1)
            # data_final.append(Fij_1)
            # data_final.append(qjk_s1)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)

            # 采用对应算法
            # model = algorithm_red.run(S1_weapon_sum, S1_target_sum, [Threat_S1], Pij_S1, Fij_1, qjk_s1, V_a, train_step)
            # 输出结果
            S1_best_plan = algorithm_red.run(S1_weapon_sum, S1_target_sum, [Threat_S1], Pij_S1, Fij_1, qjk_s1, V_a, red_step)
            # data_final.append(S1_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 对结果进行过滤
            for k, v in enumerate(S1_best_plan):
                if Fij_1[k][v] == 0:
                    S1_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(S1_best_plan):
                if v >= 0:
                    for i, j in enumerate(S1_number):
                        k = k - j
                        if k < 0:
                            S1_weapon_obe[i].manual_attack(S1_target_guid[v], S1_weapon_id[i], 1)
                            last_target_weapon.append((S1_target[v], S1_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #   print('S1第{}轮运行时间{}'.format(S1_count, end - start))
            S1_time_list.append(end - start)
            S1_count += 1
        # 武器目标分配模块
        if S2_target and S2_weapon and count % 2 == 0 or '高超声速导弹' in S2_class_num:
            start = timeit.default_timer()
            dis_S = []
            for i, j in enumerate(S2_weapon):
                dis_S = dis_S + [[float(S2_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in S2_target]] * \
                        S2_number[i]
            # 生成可行性矩阵
            Fij_S2 = feasibility(S2_weapon_range, dis_S, S2_weapon_v, S2_target_v, S2_weapon_h, S2_target_h,
                                 S2_target_name, S2_new)
            # 生成打击概率矩阵
            Pij_S2, Fij_SA2 = probability_of_hit(S2_rocket, S2_max_v, S2_max_l, S2_target_v, dis_S, S2_pof,
                                                 S2_target_a)
            Fij_2 = Fij_S2 * Fij_SA2

            s_t = []
            S2_new_weapon_obe = [str_obe[key] for key in S2_new.keys()]
            weapon_system_mapping = []
            for idx, (weapon_system, count) in enumerate(S2_new.items()):
                weapon_system_mapping.extend([idx] * count)
            for target_index, target_inf in enumerate(S2_target_inf):
                if "轰炸机" in target_inf[0]:
                    s_t.append(target_index)

            # [ 名称， 目标类型， 经度， 纬度， 方位角，打击范围， 威胁值]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(S2_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(S2_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(S2_weapon_sum)
            # data_final.append(S2_target_sum)
            # data_final.append(Threat_S2)
            # data_final.append(Pij_S2)
            # data_final.append(Fij_2)
            # data_final.append(qjk_s2)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)

            # 采用对应算法
            # model = algorithm_red.run(S2_weapon_sum, S2_target_sum, [Threat_S2], Pij_S2, Fij_2, qjk_s2, V_a, train_step)
            # 输出结果
            S2_best_plan = algorithm_red.run(S2_weapon_sum, S2_target_sum, [Threat_S2], Pij_S2, Fij_2, qjk_s2, V_a, red_step)
            # data_final.append(S2_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 对结果进行过滤
            for k, v in enumerate(S2_best_plan):
                if Fij_2[k][v] == 0:
                    S2_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(S2_best_plan):
                if v >= 0:
                    for i, j in enumerate(S2_number):
                        k = k - j
                        if k < 0:
                            S2_weapon_obe[i].manual_attack(S2_target_guid[v], S2_weapon_id[i], 1)
                            last_target_weapon.append((S2_target[v], S2_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #  print('S2第{}轮运行时间{}'.format(S2_count, end - start))
            S2_time_list.append(end - start)
            S2_count += 1
        # 武器目标分配模块
        if S3_target and S3_weapon and count % 2 == 0 or '高超声速导弹' in S3_class_num:
            start = timeit.default_timer()
            dis_S = []
            for i, j in enumerate(S3_weapon):
                dis_S = dis_S + [[float(S3_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in S3_target]] * \
                        S3_number[i]
            # 生成可行性矩阵
            Fij_S3 = feasibility(S3_weapon_range, dis_S, S3_weapon_v, S3_target_v, S3_weapon_h, S3_target_h,
                                 S3_target_name, S3_new)
            # 生成打击概率矩阵
            Pij_S3, Fij_SA3 = probability_of_hit(S3_rocket, S3_max_v, S3_max_l, S3_target_v, dis_S, S3_pof,
                                                 S3_target_a)
            Fij_3 = Fij_S3 * Fij_SA3
            S3_new_weapon_obe = [str_obe[key] for key in S3_new.keys()]
            weapon_system_mapping = []
            for idx, (weapon_system, count) in enumerate(S3_new.items()):
                weapon_system_mapping.extend([idx] * count)

            # s_t = []
            # for target_index, target_inf in enumerate(S3_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)
            #
            # [ 名称， 目标类型， 经度， 纬度， 方位角，打击范围， 威胁值]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(S3_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(S3_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(S3_weapon_sum)
            # data_final.append(S3_target_sum)
            # data_final.append(Threat_S3)
            # data_final.append(Pij_S3)
            # data_final.append(Fij_3)
            # data_final.append(qjk_s3)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)
            # 采用对应算法
            # model = algorithm_red.run(S3_weapon_sum, S3_target_sum, [Threat_S3], Pij_S3, Fij_3, qjk_s3, V_a, train_step)
            # 输出结果
            S3_best_plan = algorithm_red.run(S3_weapon_sum, S3_target_sum, [Threat_S3], Pij_S3, Fij_3, qjk_s3, V_a, red_step)

            # data_final.append(S3_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 对结果进行过滤
            for k, v in enumerate(S3_best_plan):
                if Fij_3[k][v] == 0:
                    S3_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(S3_best_plan):
                if v >= 0:
                    for i, j in enumerate(S3_number):
                        k = k - j
                        if k < 0:
                            S3_weapon_obe[i].manual_attack(S3_target_guid[v], S3_weapon_id[i], 1)
                            last_target_weapon.append((S3_target[v], S3_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #    print('S3第{}轮运行时间{}'.format(S3_count, end - start))
            S3_time_list.append(end - start)
            S3_count += 1
        # 武器目标分配模块
        if M1_target and M1_weapon and count % 5 == 0 or '高超声速导弹' in M1_class_num:
            start = timeit.default_timer()
            dis_M = []
            for i, j in enumerate(M1_weapon):
                dis_M = dis_M + [[float(M1_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in M1_target]] * \
                        M1_number[i]
            # 生成可行性矩阵
            Fij_M1 = feasibility(M1_weapon_range, dis_M, M1_weapon_v, M1_target_v, M1_weapon_h, M1_target_h,
                                 M1_target_name, M1_new)
            # 生成打击概率矩阵
            Pij_M1, Fij_MA1 = probability_of_hit(M1_rocket, M1_max_v, M1_max_l, M1_target_v, dis_M, M1_pof,
                                                 M1_target_a)
            Fij_4 = Fij_M1 * Fij_MA1
            M1_new_weapon_obe = [str_obe[key] for key in M1_new.keys()]
            weapon_system_mapping = []
            for idx, (weapon_system, count) in enumerate(M1_new.items()):
                weapon_system_mapping.extend([idx] * count)
            # s_t = []
            # for target_index, target_inf in enumerate(M1_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)
            # data = []
            # data_final = []
            # [ 武器数，目标数， 名称， 目标类型， [经度，纬度]， 方位角，打击范围， 威胁值， pij， fij， qjk, 打击结果]
            # for target_index, target in enumerate(M1_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(M1_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(M1_weapon_sum)
            # data_final.append(M1_target_sum)
            # data_final.append(Threat_M1)
            # data_final.append(Pij_M1)
            # data_final.append(Fij_4)
            # data_final.append(qjk_m1)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)

            # 采用对应算法
            # model = algorithm_red.run(M1_weapon_sum, M1_target_sum, [Threat_M1], Pij_M1, Fij_4, qjk_m1, V_a, train_step)
            # 输出结果
            M1_best_plan = algorithm_red.run(M1_weapon_sum, M1_target_sum, [Threat_M1], Pij_M1, Fij_4, qjk_m1, V_a, red_step)

            # data_final.append(M1_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 结果过滤
            for k, v in enumerate(M1_best_plan):
                if Fij_4[k][v] == 0:
                    M1_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(M1_best_plan):
                if v >= 0:
                    for i, j in enumerate(M1_number):
                        k = k - j
                        if k < 0:
                            M1_weapon_obe[i].manual_attack(M1_target_guid[v], M1_weapon_id[i], 1)
                            last_target_weapon.append((M1_target[v], M1_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #   print('M1第{}轮运行时间{}'.format(M1_count, end - start))
            M1_time_list.append(end - start)
            M1_count += 1
        # 武器目标分配模块
        if M2_target and M2_weapon and count % 5 == 0 or '高超声速导弹' in M2_class_num:
            start = timeit.default_timer()
            dis_M = []
            for i, j in enumerate(M2_weapon):
                dis_M = dis_M + [[float(M2_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in M2_target]] * \
                        M2_number[i]
            # 生成可行性矩阵
            Fij_M2 = feasibility(M2_weapon_range, dis_M, M2_weapon_v, M2_target_v, M2_weapon_h, M2_target_h,
                                 M2_target_name, M2_new)
            # 生成打击概率矩阵
            Pij_M2, Fij_MA2 = probability_of_hit(M2_rocket, M2_max_v, M2_max_l, M2_target_v, dis_M, M2_pof,
                                                 M2_target_a)
            Fij_5 = Fij_M2 * Fij_MA2
            M2_new_weapon_obe = [str_obe[key] for key in M2_new.keys()]
            weapon_system_mapping = []
            for idx, (weapon_system, count) in enumerate(M2_new.items()):
                weapon_system_mapping.extend([idx] * count)
            # s_t = []
            # for target_index, target_inf in enumerate(M2_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)
            #
            # # [ 名称， 目标类型， 经度， 纬度， 方位角，打击范围， 威胁值]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(M2_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(M2_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(M2_weapon_sum)
            # data_final.append(M2_target_sum)
            # data_final.append(Threat_M2)
            # data_final.append(Pij_M2)
            # data_final.append(Fij_5)
            # data_final.append(qjk_m2)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)

            # 采用对应算法
            # model = algorithm_red.run(M2_weapon_sum, M2_target_sum, [Threat_M2], Pij_M2, Fij_5, qjk_m2, V_a, train_step)
            # 输出结果
            M2_best_plan = algorithm_red.run(M2_weapon_sum, M2_target_sum, [Threat_M2], Pij_M2, Fij_5, qjk_m2, V_a, red_step)

            # data_final.append(M2_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 结果过滤
            for k, v in enumerate(M2_best_plan):
                if Fij_5[k][v] == 0:
                    M2_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(M2_best_plan):
                if v >= 0:
                    for i, j in enumerate(M2_number):
                        k = k - j
                        if k < 0:
                            M2_weapon_obe[i].manual_attack(M2_target_guid[v], M2_weapon_id[i], 1)
                            last_target_weapon.append((M2_target[v], M2_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #    print('M2第{}轮运行时间{}'.format(M2_count, end - start))
            M2_time_list.append(end - start)
            M2_count += 1
        # 武器目标分配模块
        if M3_target and M3_weapon and count % 5 == 0 or '高超声速导弹' in M3_class_num:
            start = timeit.default_timer()
            dis_M = []
            for i, j in enumerate(M3_weapon):
                dis_M = dis_M + [[float(M3_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in M3_target]] * \
                        M3_number[i]
            # 生成可行性矩阵
            Fij_M3 = feasibility(M3_weapon_range, dis_M, M3_weapon_v, M3_target_v, M3_weapon_h, M3_target_h,
                                 M3_target_name, M3_new)
            # 生成打击概率矩阵
            Pij_M3, Fij_MA3 = probability_of_hit(M3_rocket, M3_max_v, M3_max_l, M3_target_v, dis_M, M3_pof,
                                                 M3_target_a)
            Fij_6 = Fij_M3 * Fij_MA3
            M3_new_weapon_obe = [str_obe[key] for key in M3_new.keys()]
            weapon_system_mapping = []
            for idx, (weapon_system, count) in enumerate(M3_new.items()):
                weapon_system_mapping.extend([idx] * count)
            # s_t = []
            # for target_index, target_inf in enumerate(M3_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)
            # # [ 名称， 目标类型， 经度， 纬度， 方位角，打击范围， 威胁值]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(M3_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(M3_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(M3_weapon_sum)
            # data_final.append(M3_target_sum)
            # data_final.append(Threat_M3)
            # data_final.append(Pij_M3)
            # data_final.append(Fij_6)
            # data_final.append(qjk_m3)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)
            # 采用对应算法
            # model = algorithm_red.run(M3_weapon_sum, M3_target_sum, [Threat_M3], Pij_M3, Fij_6, qjk_m3, V_a, train_step)
            # 输出结果
            M3_best_plan = algorithm_red.run(M3_weapon_sum, M3_target_sum, [Threat_M3], Pij_M3, Fij_6, qjk_m3, V_a, red_step)

            # data_final.append(M3_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 结果过滤
            for k, v in enumerate(M3_best_plan):
                if Fij_6[k][v] == 0:
                    M3_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(M3_best_plan):
                if v >= 0:
                    for i, j in enumerate(M3_number):
                        k = k - j
                        if k < 0:
                            M3_weapon_obe[i].manual_attack(M3_target_guid[v], M3_weapon_id[i], 1)
                            last_target_weapon.append((M3_target[v], M3_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #    print('M3第{}轮运行时间{}'.format(M3_count, end - start))
            M3_time_list.append(end - start)
            M3_count += 1
        # 武器目标分配模块
        if L1_target and L1_weapon and count % 10 == 0 or '高超声速导弹' in L1_class_num:
            start = timeit.default_timer()
            dis_L = []
            for i, j in enumerate(L1_weapon):
                dis_L = dis_L + [[float(L1_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in L1_target]] * \
                        L1_number[i]
            # 生成可行性矩阵
            Fij_L1 = feasibility(L1_weapon_range, dis_L, L1_weapon_v, L1_target_v, L1_weapon_h, L1_target_h,
                                 L1_target_name, L1_new)
            # 生成打击概率矩阵
            Pij_L1, Fij_LA1 = probability_of_hit(L1_rocket, L1_max_v, L1_max_l, L1_target_v, dis_L, L1_pof,
                                                 L1_target_a)
            # 总可行性矩阵
            Fij_7 = Fij_L1 * Fij_LA1
            L1_new_weapon_obe = [str_obe[key] for key in L1_new.keys()]
            weapon_system_mapping = []
            # 根据 L1_new 字典生成拦截弹与武器系统的对应关系
            for idx, (weapon_system, count) in enumerate(L1_new.items()):
                # 将 idx 对应的值重复 count 次，idx 代表武器系统的标识符（L3 -> 0, L2 -> 1）
                weapon_system_mapping.extend([idx] * count)
            # s_t = []
            # for target_index, target_inf in enumerate(L1_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)
            # # [ 名称， 目标类型， 经度， 纬度， 方位角，打击范围， 威胁值]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(L1_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(L1_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(L1_weapon_sum)
            # data_final.append(L1_target_sum)
            # data_final.append(Threat_L1)
            # data_final.append(Pij_L1)
            # data_final.append(Fij_7)
            # data_final.append(qjk_l1)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)
            # 采用对应算法
            # model = algorithm_red.run(L1_weapon_sum, L1_target_sum, [Threat_L1], Pij_L1, Fij_7, qjk_l1, V_a, train_step)
            # 输出结果
            L1_best_plan = algorithm_red.run(L1_weapon_sum, L1_target_sum, [Threat_L1], Pij_L1, Fij_7, qjk_l1, V_a, red_step)

            # data_final.append(L1_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')
            # 结果过滤
            for k, v in enumerate(L1_best_plan):
                if Fij_7[k][v] == 0:
                    L1_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(L1_best_plan):
                if v >= 0:
                    for i, j in enumerate(L1_number):
                        k = k - j
                        if k < 0:
                            L1_weapon_obe[i].manual_attack(L1_target_guid[v], L1_weapon_id[i], 1)
                            last_target_weapon.append((L1_target[v], L1_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            #    print('L1第{}轮运行时间{}'.format(L1_count, end - start))
            L1_time_list.append(end - start)
            L1_count += 1
        if L2_target and L2_weapon and count % 10 == 0 or '高超声速导弹' in L2_class_num:
            start = timeit.default_timer()
            dis_L = []
            for i, j in enumerate(L2_weapon):
                dis_L = dis_L + [[float(L2_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in L2_target]] * \
                        L2_number[i]
            # 生成可行性矩阵
            Fij_L2 = feasibility(L2_weapon_range, dis_L, L2_weapon_v, L2_target_v, L2_weapon_h, L2_target_h,
                                 L2_target_name, L2_new)
            # 生成打击概率矩阵
            Pij_L2, Fij_LA2 = probability_of_hit(L2_rocket, L2_max_v, L2_max_l, L2_target_v, dis_L, L2_pof,
                                                 L2_target_a)
            # 总可行性矩阵
            Fij_8 = Fij_L2 * Fij_LA2
            L2_new_weapon_obe = [str_obe[key] for key in L2_new.keys()]
            weapon_system_mapping = []
            # 根据 L2_new 字典生成拦截弹与武器系统的对应关系
            for idx, (weapon_system, count) in enumerate(L2_new.items()):
                # 将 idx 对应的值重复 count 次，idx 代表武器系统的标识符（L3 -> 0, L2 -> 1）
                weapon_system_mapping.extend([idx] * count)

            # s_t = []
            # for target_index, target_inf in enumerate(L2_target_inf):
            #     if "轰炸机" in target_inf[0]:
            #         s_t.append(target_index)
            #
            # # [ 武器数，目标数， 名称， 目标类型， [经度，纬度]， 方位角，打击范围， 威胁值， ]
            # data = []
            # data_final = []
            # for target_index, target in enumerate(L2_target):
            #     data_target = dataProcess.processWtaData(target)
            #     data.append(data_target)
            # data_blue = transpose(data)
            # data_red = extract_targets_attributes(L2_new_weapon_obe)
            # data_final = data_blue + data_red
            # data_final.append(L2_weapon_sum)
            # data_final.append(L2_target_sum)
            # data_final.append(Threat_L2)
            # data_final.append(Pij_L2)
            # data_final.append(Fij_8)
            # data_final.append(qjk_l2)
            # data_final.append(V_a)
            # data_final.append(weapon_system_mapping)

            # 采用对应算法
            # model = algorithm_red.run(L2_weapon_sum, L2_target_sum, [Threat_L2], Pij_L2, Fij_8, qjk_l2, V_a, train_step)
            # 输出结果
            L2_best_plan = algorithm_red.run(L2_weapon_sum, L2_target_sum, [Threat_L2], Pij_L2, Fij_8, qjk_l2, V_a, red_step)

            # data_final.append(L2_best_plan)
            # data_final.append(fit_ness)
            # data_final.append(tour_logp)
            # # 创建 DataFrame，每个元素一列（DataFrame按列方式初始化）
            # df = pd.DataFrame([data_final])
            #
            # # Excel 文件路径
            # file_path = '数据库_sp.xlsx'
            #
            # if os.path.exists(file_path):
            #     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #         start_row = writer.sheets['Sheet1'].max_row
            #         df.to_excel(writer, index=False, header=False, startrow=start_row)
            # else:
            #     df.to_excel(file_path, index=False, engine='openpyxl')

            # 结果过滤
            for k, v in enumerate(L2_best_plan):
                if Fij_8[k][v] == 0:
                    L2_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(L2_best_plan):
                if v >= 0:
                    for i, j in enumerate(L2_number):
                        k = k - j
                        if k < 0:
                            L2_weapon_obe[i].manual_attack(L2_target_guid[v], L2_weapon_id[i], 1)
                            last_target_weapon.append((L2_target[v], L2_weapon_obe[i]))
                            break
            end = timeit.default_timer()
            # print('L2第{}轮运行时间{}'.format(L2_count, end - start))
            L2_time_list.append(end - start)
            L2_count += 1
        # print("e")

        # 战场更新及阶段统计
        scenario.mozi_server.run_grpc_simulate()
        scenario = env.step()
        # 发射的武器信息
        sam_set = red_side.get_weapons()
        # 发射武器瞄准的目标的guid的集合
        fired = [one.get_summary_info()["target"] for one in sam_set.values()]
        # target:guid plat:guid weapon:name weapon:guid
        target_weapon = [
            (one.get_summary_info()["target"], one.get_summary_info()["shooter"], one.strName, one.strGuid)
            for one in sam_set.values()]
        # 输出打击信息结果等
        for i in target_weapon:
            for j in last_target_weapon.copy():
                if i[0] == j[0].strGuid and i[1] == j[1].strGuid:
                    # print('武器平台{}向目标{}发射了{}'.format(j[1].strName, j[0].strName, i[2]))
                    # sub_plan = [平台名称, 目标名称, 武器名称, 武器guid, 目标guid, 'x']
                    sub_plan.append([j[1].strName, j[0].strName, i[2], i[3], i[0], 'x'])
                    last_target_weapon.remove(j)
        if sub_plan:
            number = number + 1
            # print('当前阶段：{}'.format(number))
            Result_plan.append(sub_plan)
        sub_plan = []

        # 监测武器和目标状态来判断是否打击成功
        for every_plan in Result_plan:
            for state in every_plan:
                w = state[3]  # 武器guide
                t = state[4]  # 目标guide
                s = state[5]  # “x"
                t_name = state[1]
                if t in target_hit:
                    state[5] = 'T'  # 已命中
                else:
                    if s == 'x':
                        if not scenario.unit_is_alive(w):
                            if scenario.unit_is_alive(t):
                                state[5] = 'F'  # 武器没了目标还在 -》失败
                            else:
                                state[5] = 'T'  # 武器和目标都没了 -》命中
                                target_hit.append(t)
                                target_hit_name.append(t_name)
                                count_hit += 1

        # 根据发射的武器确定取消积压命令和避免浪费
        if last_target_weapon:
            for i in last_target_weapon:
                t, w = i[0].strGuid, i[1].strGuid
                if t in S1_target_guid:
                    index = S1_target_guid.index(t)
                    weapon_index = S1_best_plan.index(index)
                    for k, v in enumerate(S1_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and S1_weapon_guid[k] == w:
                            S1_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in S2_target_guid:
                    index = S2_target_guid.index(t)
                    weapon_index = S2_best_plan.index(index)
                    for k, v in enumerate(S2_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and S2_weapon_guid[k] == w:
                            S2_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in S3_target_guid:
                    index = S3_target_guid.index(t)
                    weapon_index = S3_best_plan.index(index)
                    for k, v in enumerate(S3_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and S3_weapon_guid[k] == w:
                            S3_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in M1_target_guid:
                    index = M1_target_guid.index(t)
                    weapon_index = M1_best_plan.index(index)
                    for k, v in enumerate(M1_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and M1_weapon_guid[k] == w:
                            M1_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in M2_target_guid:
                    index = M2_target_guid.index(t)
                    weapon_index = M2_best_plan.index(index)
                    for k, v in enumerate(M2_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and M2_weapon_guid[k] == w:
                            M2_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in M3_target_guid:
                    index = M3_target_guid.index(t)
                    weapon_index = M3_best_plan.index(index)
                    for k, v in enumerate(M3_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and M3_weapon_guid[k] == w:
                            M3_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in L1_target_guid:
                    index = L1_target_guid.index(t)
                    weapon_index = L1_best_plan.index(index)
                    for k, v in enumerate(L1_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and L1_weapon_guid[k] == w:
                            L1_weapon_obe[k].unit_drop_target_contact(t)
                            break
                elif t in L2_target_guid:
                    index = L2_target_guid.index(t)
                    weapon_index = L2_best_plan.index(index)
                    for k, v in enumerate(L2_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and L2_weapon_guid[k] == w:
                            L2_weapon_obe[k].unit_drop_target_contact(t)
                            break
        time = scenario.m_Duration.split('@')
        duration = int(time[0]) * 86400 + int(time[1]) * 3600 + int(time[2]) * 60
        if scenario.m_StartTime + duration <= scenario.m_Time:
            Time_list = [S1_time_list, S2_time_list, S3_time_list, M1_time_list, M2_time_list,
                         M3_time_list, L1_time_list, L2_time_list, T_time_list,
                         [1 - Des, count_hit / len(TARGET), count_hit, len(TARGET)]]
            name = str('A_name') + '_' + str('N') + '.csv'
            for i in Time_list:
                with open(name, 'a', newline='') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(i)
            print('推演已结束！')
            red_facilities_in = red_side.get_facilities().values()
            damage_list_1 = get_red_damage(red_facilities_in)
            damage_vate = dataProcess.compute_damage(damage_list_0, damage_list_1)
            r1 = damage_vate
            r2 = count_hit / len(TARGET)
            reward_destory_blue = 1
            reward_damaged_blue = 2
            break
            # sys.exit(0)
        else:
            pass

    return r1, r2, algorithm_red, reward_destory_blue, reward_damaged_blue, algorithm_blue, data_train_blue


def main():
    """主函数"""
    args = parser.parse_args()
    if args.platform_mode == 'versus':
        print('比赛模式')
        ip_port = args.avail_ip_port.split(':')
        ip = ip_port[0]
        port = ip_port[1]
        env = Environment(ip, port, duration_interval=etc.DURATION_INTERVAL, app_mode=2,
                          agent_key_event_file=args.agent_key_event_file, platform_mode=args.platform_mode)

    else:
        # 红方训练
        red_step = 0
        blue_step = 0
        reward1_line = []
        reward2_line = []
        for j in range(20):
            print('开发模式')
            env = Environment(ip=etc.SERVER_IP, port=etc.SERVER_PORT, platform=etc.PLATFORM,
                              scenario_name=etc.SCENARIO_NAME, simulate_compression=etc.SIMULATE_COMPRESSION,
                              duration_interval=etc.DURATION_INTERVAL, synchronous=etc.SYNCHRONOUS,
                              app_mode=etc.app_mode)

            reward1, reward2, algorithm_red, reward_destory_blue, reward_damaged_blue, algorithm_blue, data_train_blue = run(env, blue_step, red_step)
            red_step += 1
            algorithm_red.train_pointer(reward1, reward2)
            reward1_line.append(1 - reward1)
            reward2_line.append(reward2)
            print(f'reward1 = ', reward1_line)
            print(f'reward2 = ', reward2_line)
        print(f'reward1 = ',reward1_line)
        print(f'reward2 = ',reward2_line)
        print(f'reward_line=', reward_line)
        for j in range(3):
            print('开发模式')
            env = Environment(ip=etc.SERVER_IP, port=etc.SERVER_PORT, platform=etc.PLATFORM,
                              scenario_name=etc.SCENARIO_NAME, simulate_compression=etc.SIMULATE_COMPRESSION,
                              duration_interval=etc.DURATION_INTERVAL, synchronous=etc.SYNCHRONOUS,
                              app_mode=etc.app_mode)

            reward1, reward2, algorithm_red, reward_destory_blue, reward_damaged_blue, algorithm_blue, data_train_blue = run(env, blue_step, red_step)
            blue_step += 1
            algorithm_blue.train_pointer(reward_destory_blue, reward_damaged_blue, data_train_blue)


    return


try:
    main()
except Exception as e:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'error_{timestamp}.log'
    with open(log_filename, 'w', encoding='utf-8') as error_file:
        traceback.print_exc(file=error_file)
    sys.exit()
