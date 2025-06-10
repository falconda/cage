import os
import sys
import argparse
import random
import numpy as np
# from WTA.TS_WTA import TS
# from WTA.GA_WTA import GA
# from WTA.ABC_WTA import ABC
# from WTA.ACO_WTA import ACO
# from WTA.ICA_WTA import ICA
# from WTA.PSO_GA_WTA import PSO_GA
# from WTA.PSO_WTA import PSO
# from WTA.SFLA_WTA import SFLA
# from WTA.IGA_WTA import IGA

from WNN import WNN_TA
from mozi_ai_sdk.FKFD.env.env import Environment
from mozi_ai_sdk.FKFD.env import etc
from Functions import feasibility, probability_of_hit, get_target_A, get_weapon_set, get_current_num, weapon_info, \
    get_class_num

sys.path.append('D:\\workplace\\moziai\\mozi_ai_sdk\\FKFD\\WTA\\RNN')
from RNN_WTA import RNN_WT

parser = argparse.ArgumentParser()
parser.add_argument("--avail_ip_port", type=str, default='127.0.0.1:6060')
parser.add_argument("--platform_mode", type=str, default='eval')
parser.add_argument("--side_name", type=str, default='红方')
parser.add_argument("--agent_key_event_file", type=str, default=None)

#  设置墨子安装目录下bin目录为MOZIPATH，程序会自动启动墨子
os.environ['MOZIPATH'] = 'D:\\programfiles\\mozi\\Mozi\\MoziServer\\bin'
print(os.environ['MOZIPATH'])


# run函数
def run(env, side_name=None):
    if not side_name:
        side_name = '红方'
    # 启动墨子服务器，连接墨子服务器，获取初始态势数据
    env.start()
    # 加载想定，初始化推演方
    env.reset()
    # 获取更新态势
    scenario = env.step()
    # 获取推演方，获取本方的所有数据
    # (返回的是本方的所有数据：对应API中的CSide)
    red_side = scenario.get_side_by_name(side_name)
    print('进入推演方%s' % side_name)
    # 获得敌方的所有数据
    # (返回目标的字典{guid:obe.....})
    contacts_dic = red_side.contacts
    # 推演方条令
    temp = [i for i in contacts_dic.values()]
    for i in temp:
        i.set_mark_contact('H')
    # 准静态化更新
    red_side.static_construct()
    facilities = red_side.get_facilities()
    # 本方的武器初始化可知
    # 基地1的近程武器平台
    B1 = [facility for facility in facilities.values() if '基地1' in facility.strName][0]
    S1 = [facility for facility in facilities.values() if 'S1-1' in facility.strName][0]
    S2 = [facility for facility in facilities.values() if 'S1-2' in facility.strName][0]
    S3 = [facility for facility in facilities.values() if 'S1-3' in facility.strName][0]
    # 基地1的中远程武器平台
    M1 = [facility for facility in facilities.values() if 'M1-1' in facility.strName][0]
    M2 = [facility for facility in facilities.values() if 'M1-2' in facility.strName][0]
    # 基地1的远程武器平台
    L1 = [facility for facility in facilities.values() if 'L1' in facility.strName][0]
    # 基地2的近程武器平台
    B2 = [facility for facility in facilities.values() if '基地2' in facility.strName][0]
    S11 = [facility for facility in facilities.values() if 'S2-1' in facility.strName][0]
    S22 = [facility for facility in facilities.values() if 'S2-2' in facility.strName][0]
    S33 = [facility for facility in facilities.values() if 'S2-3' in facility.strName][0]
    # 基地2的中远程武器平台
    M11 = [facility for facility in facilities.values() if 'M2-1' in facility.strName][0]
    M22 = [facility for facility in facilities.values() if 'M2-2' in facility.strName][0]
    # 基地2的远程武器平台
    L2 = [facility for facility in facilities.values() if 'L2' in facility.strName][0]
    # 基地3的近程武器平台
    B3 = [facility for facility in facilities.values() if '基地3' in facility.strName][0]
    S111 = [facility for facility in facilities.values() if 'S3-1' in facility.strName][0]
    S222 = [facility for facility in facilities.values() if 'S3-2' in facility.strName][0]
    S333 = [facility for facility in facilities.values() if 'S3-3' in facility.strName][0]
    # 基地3的中远程武器平台
    M111 = [facility for facility in facilities.values() if 'M3-1' in facility.strName][0]
    M222 = [facility for facility in facilities.values() if 'M3-2' in facility.strName][0]
    L3 = [facility for facility in facilities.values() if 'L3' in facility.strName][0]
    SL = [facility for facility in facilities.values() if 'S-400E' in facility.strName][0]
    S300_1 = [facility for facility in facilities.values() if 'S300-1' in facility.strName][0]
    S300_2 = [facility for facility in facilities.values() if 'S300-2' in facility.strName][0]
    TH1 = [facility for facility in facilities.values() if 'TH1' in facility.strName][0]
    TH2 = [facility for facility in facilities.values() if 'TH2' in facility.strName][0]
    GMD1 = [facility for facility in facilities.values() if 'GMD1' in facility.strName][0]
    GMD2 = [facility for facility in facilities.values() if 'GMD2' in facility.strName][0]
    # 挂架；武器编号；射程；目标高度；目标速度；名称；导弹动力系数；基础命中率；挂架上导弹数；初始总数
    str_obe = {'S1': S1, 'S2': S2, 'S3': S3, 'M1': M1, 'M2': M2, 'L1': L1,
               'S11': S11, 'S22': S22, 'S33': S33, 'M11': M11, 'M22': M22, 'L2': L2,
               'S111': S111, 'S222': S222, 'S333': S333, 'M111': M111, 'M222': M222, 'L3': L3,
               'SL': SL, 'S300_1': S300_1, 'S300_2': S300_2,
               'TH1': TH1, 'TH2': TH2, 'GMD1': GMD1, 'GMD2': GMD2}
    every_weapon_mounts = {'S1': 8, 'S2': 8, 'S3': 8, 'M1': 12, 'M2': 12, 'L1': 12,
                           'S11': 8, 'S22': 8, 'S33': 8, 'M11': 12, 'M22': 12, 'L2': 12,
                           'S111': 8, 'S222': 8, 'S333': 8, 'M111': 12, 'M222': 12, 'L3': 12,
                           'SL': 16, 'S300_1': 8, 'S300_2': 8,
                           'TH1': 6, 'TH2': 6,
                           'GMD1': 24, 'GMD2': 24}
    target_a = {'干扰机': 1, '预警机': 1, '女武神无人机': 4, 'F-16DJ': 4.9, 'B-1B轰炸机': 1, 'B-52H轰炸机': 1.5, 'RQ-180': 1,
                'F-15E战斗轰炸机': 4.5, '超级眼镜蛇直升机': 2, 'F-16CM战斗机': 4.9, '枪骑兵轰炸机': 2, '巡飞弹': 1, '高超声速导弹': 0, '导弹': 0,
                '炸弹': 0}
    target_name = ['干扰机', '预警机', '女武神无人机', 'F-16DJ', 'B-1B轰炸机', 'B-52H轰炸机', 'RQ-180', 'F-15E战斗轰炸机', '超级眼镜蛇直升机',
                   'F-16CM战斗机', '枪骑兵轰炸机', '巡飞弹', '高超声速导弹', '导弹', '炸弹']
    special_target = ['高超声速导弹']
    special_target_L = ['超级眼镜蛇直升机', 'F-16DJ', '高超声速导弹']
    global S1_weapon_obe, S1_max_r, S1_weapon_guid, S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
        S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_sum, S1_weapon_every, S1_init_num, S1_target_v, S1_target_h, S1_target_a, Threat_S1, qjk_s1, S1_target_sum, S2_target_v, S2_target_h, S2_target_a, S2_target_sum, Threat_S2, qjk, qjk_s2, S3_target_v, S3_target_h, S3_target_a, S3_target_sum, Threat_S3, qjk_s3, M1_target_v, M1_target_h, M1_target_a, M1_target_sum, Threat_M1, qjk_m1, M2_target_v, M2_target_h, M2_target_a, M2_target_sum, Threat_M2, M3_target_v, M3_target_h, M3_target_a, M3_target_sum, Threat_M3, qjk_m3, L1_target_v, L1_target_h, L1_target_a, L1_target_sum, Threat_L1, qjk_l1, qjk_m2
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
    T_target_guid, T_best_plan, T_no_storage = [], [], []
    count = 0
    time_count = 210
    NET = 1
    S1_weapon = []
    S2_weapon = []
    S3_weapon = []
    M1_weapon = []
    M2_weapon = []
    M3_weapon = []
    L1_weapon = []
    L2_weapon = []
    T_weapon = []
    S1_weapon_guid, S2_weapon_guid, S3_weapon_guid, M1_weapon_guid = [], [], [], []
    M2_weapon_guid, M3_weapon_guid, L1_weapon_guid, L2_weapon_guid = [], [], [], []
    T_weapon_guid = []
    # 近程范围固有武器
    S1_original_weapon = ['S1', 'S2', 'S3']
    S2_original_weapon = ['S11', 'S22', 'S33']
    S3_original_weapon = ['S111', 'S222', 'S333']
    # 各区域候选武器平台
    S1_prepare = ['S300_1', 'L1', 'M1', 'SL']
    S2_prepare = ['S300_1', 'S300_2', 'M1', 'M111', 'L1', 'L2', 'L3', 'SL']
    S3_prepare = ['S300_2', 'L3', 'M111', 'SL']
    M1_prepare = ['S300_1', 'L1', 'L2', 'L3', 'M11', 'SL']
    M2_prepare = ['L1', 'L2', 'L3', 'M1', 'M111', 'M2', 'M222', 'S300_1', 'S300_2', 'SL']
    M3_prepare = ['S300_2', 'L1', 'L2', 'L3', 'M22', 'SL']
    L1_prepare = ['SL', 'S300_1']
    L2_prepare = ['SL', 'S300_2']
    All_weapon = ['L1', 'L2', 'L3', 'M1', 'M2', 'M11', 'M22', 'M111', 'M222', 'S1', 'S2',
                  'S3', 'S11', 'S22', 'S33', 'S111', 'S222', 'S333', 'S300_1', 'S300_2', 'SL',
                  'TH1', 'TH2', 'GMD1', 'GMD2']
    All_weapon_obe = [L1, L2, L3, M1, M2, M11, M22, M111, M222, S1, S2,
                      S3, S11, S22, S33, S111, S222, S333, S300_1, S300_2, SL, TH1, TH2, GMD1, GMD2]
    All_weapon_guid = [obe.strGuid for obe in All_weapon_obe]
    Base_obe = [B1, B2, B3]
    Base_guid = [i.strGuid for i in Base_obe]
    Des_base = [0, 0, 0]
    # 基地或建筑的价值
    V_a = [[80, 85, 90]]
    # 消除的目标数
    count_hit = 0
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

    def update_weapon_T():
        global T_weapon_obe, T_max_r, T_weapon_guid, T_number, T_weapon_id, T_weapon_name, T_weapon_range, T_weapon_h, \
            T_weapon_v, T_rocket, T_max_v, T_max_l, T_pof, T_weapon_sum, T_weapon_every, T_init_num
        if T_weapon:
            T_weapon_obe = [str_obe[i] for i in T_weapon]
            T_weapon_guid = [obe.strGuid for obe in T_weapon_obe]
            T_number, T_weapon_id, T_weapon_name, T_weapon_range, T_weapon_h, \
            T_weapon_v, T_rocket, T_max_v, T_max_l, T_pof, T_weapon_every, T_init_num, T_max_r = get_weapon_set(
                T_weapon, weapon_info)
            T_weapon_sum = sum(T_number)
        else:
            T_weapon_obe = []
            T_weapon_guid = []
    # 暂存发射的武器和目标
    Result_plan = []
    # 每一轮的打击计划
    sub_plan = []
    # 暂不可用武器平台
    Temp_unused = []
    # 总的目标数量
    TARGET = []
    new_mounts = every_weapon_mounts.copy()
    while True:
        while time_count > 0:
            env.step()
            time_count -= 1
        count = count + 1
        for k, v in enumerate(Base_guid):
            if red_side.get_unit_by_guid(v) is None:
                Des_base[k] = 1 * V_a[0][k]
            else:
                Des_base[k] = 0.01 * float(Base_obe[k].strDamageState) * V_a[0][k]

        Des = sum(Des_base) / sum(V_a[0])
        print('基地损失价值百分比:' + str(Des))
        # 选择算法
        Al = RNN_WT
        # 目标的对象
        # start = timeit.default_timer()
        contacts_dic = red_side.contacts
        # 目标速度大于5判定为空中目标
        targets = [item for item in contacts_dic.values() if item.fCurrentSpeed > 5 and '蜂群无人机' not in item.strName]
        print('当前目标数量:{}'.format(len(targets)))
        targets_copy = targets.copy()
        targets_guid = [T.strGuid for T in targets]
        TARGET = TARGET + targets_guid
        TARGET = list(set(TARGET))
        print('总的目标数量:{}'.format(len(TARGET)))
        print('消灭的目标数量:{}'.format(count_hit))
        # 已被分配武器的目标不作为下一阶段打击的目标
        for g in fired:
            if g in targets_guid:
                index = targets_guid.index(g)
                temp_t = targets_copy[index]
                if temp_t in targets:
                    targets.remove(temp_t)
        S1_target, S2_target, S3_target, M1_target, M2_target, M3_target, L1_target, L2_target, T_target = [], [], [], [], [], [], [], [], []
        # 进行目标的划分
        for target in targets:
            # 目标纬度
            latitude = float(target.dLatitude)
            # 目标经度
            longitude = float(target.dLongitude)
            if '弹道导弹' in target.strName or '核弹炸药' in target.strName:
                T_target.append(target)
            elif longitude >= 119.6:
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
        # 各防空区域目标的guid
        S1_target_guid = [target.strGuid for target in S1_target]
        S2_target_guid = [target.strGuid for target in S2_target]
        S3_target_guid = [target.strGuid for target in S3_target]
        M1_target_guid = [target.strGuid for target in M1_target]
        M2_target_guid = [target.strGuid for target in M2_target]
        M3_target_guid = [target.strGuid for target in M3_target]
        L1_target_guid = [target.strGuid for target in L1_target]
        L2_target_guid = [target.strGuid for target in L2_target]
        T_target_guid = [target.strGuid for target in T_target]
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
        T_weapon = []
        Temp_unused = []
        # 记录打击的目标
        last_target_weapon = []
        T_target_sum, Threat_T, qjk_T, T_target_v, T_target_h, T_target_name, T_target_a, T_class_num = None, [], None, None, None, None, None, {}
        if T_target:
            # 目标数量
            T_target_sum = len(T_target)
            # 各目标速度：list
            T_target_v = [target.fCurrentSpeed for target in T_target]
            # 各目标高度：list
            T_target_h = [target.fCurrentAltitude_ASL for target in T_target]
            # 目标名字
            T_target_name = [target.strName for target in T_target]
            # 目标机动系数
            T_target_a, T_target_class = get_target_A(T_target_name, target_name, target_a)
            T_class_num = get_class_num(T_target_class)
            # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
            T_target_inf = [
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
                 target.fCurrentHeading] for target in T_target]
            Threat_T, Damage_T, Base_T = WNN_TA(T_target_inf).run()
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, T_target_sum])[0]
            qjk_T = np.zeros((T_target_sum, 3))
            for k, v in enumerate(Base_T):
                qjk_T[k][v] = Damage_T[k]

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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
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
                [target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
                 target.fCurrentHeading] for target in L1_target]
            Threat_L1, Damage_L1, Base_L1 = WNN_TA(L1_target_inf).run()
            # 生成目标威胁值
            #Threat_L1 = [1 for d in range(1, L1_target_sum + 1)]
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
            L2_target_inf = [[target.strName, target.fCurrentSpeed, target.fCurrentAltitude_ASL, target.dLongitude, target.dLatitude,
                              target.fCurrentHeading] for target in L2_target]
            Threat_L2, Damage_L2, Base_L2 = WNN_TA(L2_target_inf).run()
            # 生成目标威胁值
            #Threat_L2 = [1 for d in range(1, L2_target_sum + 1)]
            # 目标打击基地的概率值
            # T_to_A = np.random.randint(0, 2, [1, L2_target_sum])[0]
            qjk_l2 = np.zeros((L2_target_sum, 3))
            for k, v in enumerate(Base_L2):
                qjk_l2[k][v] = Damage_L2[k]
        # 根据各区域的威胁、数量、种类等分配合适的武器
        # 根据特殊目标分配特殊武器,并执行本地任务
        # new_mounts = every_weapon_mounts.copy()
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
                    # 目标数量过多，后期需要请求调用别处资源
                    if M1_target_norm >= num_m1 + num_m2:
                        M1_new['M1'] = num_m1
                        M1_new['M2'] = num_m2
                        new_mounts['M1'] = 0
                        new_mounts['M2'] = 0
                        M1_target_norm = M1_target_norm - num_m1 - num_m2
                    # 目标数量小，可剩余资源供后期别处使用
                    else:
                        error = num_m1 - M1_target_norm
                        # 如果M1够用
                        if error > 0:
                            M1_new['M1'] = M1_target_norm
                            new_mounts['M1'] = error
                        # 如果不够用
                        else:
                            M1_new['M1'] = num_m1
                            new_mounts['M1'] = 0
                            M1_new['M2'] = -error
                            new_mounts['M2'] = num_m2 + error
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
                    # 目标数量过多，后期需要请求调用别处资源
                    if M2_target_norm >= num_m1 + num_m2:
                        M2_new['M11'] = num_m1
                        M2_new['M22'] = num_m2
                        new_mounts['M11'] = 0
                        new_mounts['M22'] = 0
                        M2_target_norm = M2_target_norm - num_m1 - num_m2
                    # 目标数量小，可剩余资源供后期别处使用
                    else:
                        error = num_m1 - M2_target_norm
                        # 如果M1够用
                        if error > 0:
                            M2_new['M11'] = M2_target_norm
                            new_mounts['M11'] = error
                        # 如果不够用
                        else:
                            M2_new['M11'] = num_m1
                            new_mounts['M11'] = 0
                            M2_new['M22'] = -error
                            new_mounts['M22'] = num_m2 + error
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
                    # 目标数量小，可剩余资源供后期别处使用
                    else:
                        error = num_m1 - M3_target_norm
                        # 如果M1够用
                        if error > 0:
                            M3_new['M111'] = M3_target_norm
                            new_mounts['M111'] = error
                        # 如果不够用
                        else:
                            M3_new['M111'] = num_m1
                            new_mounts['M111'] = 0
                            M3_new['M222'] = -error
                            new_mounts['M222'] = num_m2 + error
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
                    # 目标数量小，可剩余资源供后期别处使用
                    else:
                        error = num_m1 - L1_target_norm
                        # 如果M1够用
                        if error > 0:
                            L1_new['L1'] = L1_target_norm
                            new_mounts['L1'] = error
                        # 如果不够用
                        else:
                            L1_new['L1'] = num_m1
                            new_mounts['L1'] = 0
                            L1_new['L2'] = -error
                            new_mounts['L2'] = num_m2 + error
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
                    # 目标数量小，可剩余资源供后期别处使用
                    else:
                        error = num_m1 - L2_target_norm
                        # 如果M1够用
                        if error > 0:
                            L2_new['L3'] = L2_target_norm
                            new_mounts['L3'] = error
                        # 如果不够用
                        else:
                            L2_new['L3'] = num_m1
                            new_mounts['L3'] = 0
                            L2_new['L2'] = -error
                            new_mounts['L2'] = num_m2 + error
                        L2_target_norm = 0
            # 执行支援任务(近程、中程、远程，内部由威胁度决定顺序)
            A_target_norm = [S1_target_norm, S2_target_norm, S3_target_norm, M1_target_norm, M2_target_norm, M3_target_norm,
                             L1_target_norm, L2_target_norm]
            A_prepare = [S1_prepare, S2_prepare, S3_prepare, M1_prepare, M2_prepare, M3_prepare, L1_prepare, L2_prepare]
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
            # print(S1_new)
            # print(S2_new)
            # print(S3_new)
            # print(M1_new)
            # print(M2_new)
            # print(M3_new)
            # print(L1_new)
            # print(L2_new)
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
            S1_weapon_old = ['S1', 'S2', 'S3', 'S300_1']
            for i in S1_weapon_old:
                if new_mounts[i] > 0:
                    S1_new[i] = new_mounts[i]
                    S1_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_S1()
            S2_new = {}
            S2_weapon_old = ['S11', 'S22', 'S33']
            for i in S2_weapon_old:
                if new_mounts[i] > 0:
                    S2_new[i] = new_mounts[i]
                    S2_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_S2()
            S3_new = {}
            S3_weapon_old = ['S111', 'S222', 'S333', 'S300_2']
            for i in S3_weapon_old:
                if new_mounts[i] > 0:
                    S3_new[i] = new_mounts[i]
                    S3_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_S3()
            M1_new = {}
            M1_weapon_old = ['M1', 'M2']
            for i in M1_weapon_old:
                if new_mounts[i] > 0:
                    M1_new[i] = new_mounts[i]
                    M1_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_M1()
            M2_new = {}
            M2_weapon_old = ['M11', 'M22']
            for i in M2_weapon_old:
                if new_mounts[i] > 0:
                    M2_new[i] = new_mounts[i]
                    M2_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            update_weapon_M2()
            M3_new = {}
            M3_weapon_old = ['M11', 'M22']
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
        T_new = {}
        T_weapon_old = ['TH1', 'TH2', 'GMD1', 'GMD2']
        for i in T_weapon_old:
            if new_mounts[i] > 0:
                T_new[i] = new_mounts[i]
                T_weapon.append(i)
                weapon_info[i]['num_weapon'] = new_mounts[i]
        update_weapon_T()
        if T_target and T_weapon:
            dis_S = []
            for i, j in enumerate(T_weapon):
                dis_S = dis_S + [[float(T_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in T_target]] * \
                        T_number[i]
            # 生成可行性矩阵
            Fij_T = feasibility(T_weapon_range, dis_S, T_weapon_v, T_target_v, T_weapon_h, T_target_h,
                                T_target_name, T_new)
            # 生成打击概率矩阵
            Pij_T, Fij_SA1 = probability_of_hit(T_rocket, T_max_v, T_max_l, T_target_v, dis_S, T_pof,
                                                T_target_a, 0)
            Fij_1 = Fij_T * Fij_SA1
            # 采用对应算法
            model = Al(T_weapon_sum, T_target_sum, [Threat_T], Pij_T, Fij_1, qjk_T, V_a)
            # 输出结果
            T_best_plan = model.run()
            # 对结果进行过滤
            for k, v in enumerate(T_best_plan):
                if Fij_1[k][v] == 0:
                    T_best_plan[k] = -1
            # 武器平台进行打击
            for k, v in enumerate(T_best_plan):
                if v >= 0:
                    for i, j in enumerate(T_number):
                        k = k - j
                        if k < 0:
                            T_weapon_obe[i].manual_attack(T_target_guid[v], T_weapon_id[i], 1)
                            last_target_weapon.append((T_target[v], T_weapon_obe[i]))
                            break
        # 武器目标分配模块
        if (S1_target and S1_weapon) and count % 2 == 0 or '高超声速导弹' in S1_class_num:
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
            # 采用对应算法
            model = Al(S1_weapon_sum, S1_target_sum, [Threat_S1], Pij_S1, Fij_1, qjk_s1, V_a)
            # 输出结果
            S1_best_plan = model.run()
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
        # 武器目标分配模块
        if S2_target and S2_weapon and count % 2 == 0 or '高超声速导弹' in S2_class_num:
            dis_S = []
            for i, j in enumerate(S2_weapon):
                dis_S = dis_S + [[float(S2_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in S2_target]] * \
                        S2_number[i]
            # 生成可行性矩阵
            Fij_S2 = feasibility(S2_weapon_range, dis_S, S2_weapon_v, S2_target_v, S2_weapon_h, S2_target_h,
                                 S2_target_name, S2_new)
            # 生成打击概率矩阵
            Pij_S2, Fij_SA2 = probability_of_hit(S2_rocket, S2_max_v, S2_max_l, S2_target_v, dis_S, S2_pof, S2_target_a)
            Fij_2 = Fij_S2 * Fij_SA2

            # 采用对应算法
            model = Al(S2_weapon_sum, S2_target_sum, [Threat_S2], Pij_S2, Fij_2, qjk_s2, V_a)
            # 输出结果
            S2_best_plan = model.run()
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
        # 武器目标分配模块
        if S3_target and S3_weapon and count % 2 == 0 or '高超声速导弹' in S3_class_num:
            dis_S = []
            for i, j in enumerate(S3_weapon):
                dis_S = dis_S + [[float(S3_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in S3_target]] * \
                        S3_number[i]
            # 生成可行性矩阵
            Fij_S3 = feasibility(S3_weapon_range, dis_S, S3_weapon_v, S3_target_v, S3_weapon_h, S3_target_h,
                                 S3_target_name, S3_new)
            # 生成打击概率矩阵
            Pij_S3, Fij_SA3 = probability_of_hit(S3_rocket, S3_max_v, S3_max_l, S3_target_v, dis_S, S3_pof, S3_target_a)
            Fij_3 = Fij_S3 * Fij_SA3
            # 采用对应算法
            model = Al(S3_weapon_sum, S3_target_sum, [Threat_S3], Pij_S3, Fij_3, qjk_s3, V_a)
            # 输出结果
            S3_best_plan = model.run()
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
        # 武器目标分配模块
        if M1_target and M1_weapon and count % 5 == 0 or '高超声速导弹' in M1_class_num:
            dis_M = []
            for i, j in enumerate(M1_weapon):
                dis_M = dis_M + [[float(M1_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in M1_target]] * \
                        M1_number[i]
            # 生成可行性矩阵
            Fij_M1 = feasibility(M1_weapon_range, dis_M, M1_weapon_v, M1_target_v, M1_weapon_h, M1_target_h,
                                 M1_target_name, M1_new)
            # 生成打击概率矩阵
            Pij_M1, Fij_MA1 = probability_of_hit(M1_rocket, M1_max_v, M1_max_l, M1_target_v, dis_M, M1_pof, M1_target_a)
            Fij_4 = Fij_M1 * Fij_MA1
            # 采用对应算法
            model = Al(M1_weapon_sum, M1_target_sum, [Threat_M1], Pij_M1, Fij_4, qjk_m1, V_a)
            # 输出结果
            M1_best_plan = model.run()
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
        # 武器目标分配模块
        if M2_target and M2_weapon and count % 5 == 0 or '高超声速导弹' in M2_class_num:
            dis_M = []
            for i, j in enumerate(M2_weapon):
                dis_M = dis_M + [[float(M2_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in M2_target]] * \
                        M2_number[i]
            # 生成可行性矩阵
            Fij_M2 = feasibility(M2_weapon_range, dis_M, M2_weapon_v, M2_target_v, M2_weapon_h, M2_target_h,
                                 M2_target_name, M2_new)
            # 生成打击概率矩阵
            Pij_M2, Fij_MA2 = probability_of_hit(M2_rocket, M2_max_v, M2_max_l, M2_target_v, dis_M, M2_pof, M2_target_a)
            Fij_5 = Fij_M2 * Fij_MA2
            # 采用对应算法
            model = Al(M2_weapon_sum, M2_target_sum, [Threat_M2], Pij_M2, Fij_5, qjk_m2, V_a)
            # 输出结果
            M2_best_plan = model.run()
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
        # 武器目标分配模块
        if M3_target and M3_weapon and count % 5 == 0 or '高超声速导弹' in M3_class_num:
            dis_M = []
            for i, j in enumerate(M3_weapon):
                dis_M = dis_M + [[float(M3_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in M3_target]] * \
                        M3_number[i]
            # 生成可行性矩阵
            Fij_M3 = feasibility(M3_weapon_range, dis_M, M3_weapon_v, M3_target_v, M3_weapon_h, M3_target_h,
                                 M3_target_name, M3_new)
            # 生成打击概率矩阵
            Pij_M3, Fij_MA3 = probability_of_hit(M3_rocket, M3_max_v, M3_max_l, M3_target_v, dis_M, M3_pof, M3_target_a)
            Fij_6 = Fij_M3 * Fij_MA3
            # 采用对应算法
            model = Al(M3_weapon_sum, M3_target_sum, [Threat_M3], Pij_M3, Fij_6, qjk_m3, V_a)
            # 输出结果
            M3_best_plan = model.run()
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
        # 武器目标分配模块
        if L1_target and L1_weapon and count % 10 == 0 or '高超声速导弹' in L1_class_num:
            dis_L = []
            for i, j in enumerate(L1_weapon):
                dis_L = dis_L + [[float(L1_weapon_obe[i].get_range_to_contact(T.strGuid)) for T in L1_target]] * \
                        L1_number[i]
            # 生成可行性矩阵
            Fij_L1 = feasibility(L1_weapon_range, dis_L, L1_weapon_v, L1_target_v, L1_weapon_h, L1_target_h,
                                 L1_target_name, L1_new)
            # 生成打击概率矩阵
            Pij_L1, Fij_LA1 = probability_of_hit(L1_rocket, L1_max_v, L1_max_l, L1_target_v, dis_L, L1_pof, L1_target_a)
            # 总可行性矩阵
            Fij_7 = Fij_L1 * Fij_LA1
            # 采用对应算法
            model = Al(L1_weapon_sum, L1_target_sum, [Threat_L1], Pij_L1, Fij_7, qjk_l1, V_a)
            # 输出结果
            L1_best_plan = model.run()
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
        if L2_target and L2_weapon and count % 10 == 0 or '高超声速导弹' in L2_class_num:
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
            # 采用对应算法
            model = Al(L2_weapon_sum, L2_target_sum, [Threat_L2], Pij_L2, Fij_8, qjk_l2, V_a)
            # 输出结果
            L2_best_plan = model.run()
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
            # end = timeit.default_timer()
            # print('Runtime:{}'.format(end - start))
        # 战场更新及阶段统计
        scenario.mozi_server.run_grpc_simulate()
        scenario = env.step()
        # 发射的武器信息
        sam_set = red_side.get_weapons()
        # 发射武器瞄准的目标的guid的集合
        fired = [one.get_summary_info()["target"] for one in sam_set.values()]
        # target:guid plat:guid weapon:name weapon:guid
        target_weapon = [(one.get_summary_info()["target"], one.get_summary_info()["shooter"], one.strName, one.strGuid)
                         for one in sam_set.values()]
        # 输出打击信息结果等
        for i in target_weapon:
            for j in last_target_weapon.copy():
                if i[0] == j[0].strGuid and i[1] == j[1].strGuid:
                    print('武器平台{}向目标{}发射了{}'.format(j[1].strName, j[0].strName, i[2]))
                    sub_plan.append([j[1].strName, j[0].strName, i[2], i[3], i[0], 'x'])
                    last_target_weapon.remove(j)
        if sub_plan:
            number = number + 1
            print('当前阶段：{}'.format(number))
            Result_plan.append(sub_plan)
        sub_plan = []
        # 监测武器和目标状态来判断是否打击成功
        for every_plan in Result_plan:
            for state in every_plan:
                w = state[3]
                t = state[4]
                s = state[5]
                if s == 'x':
                    if not scenario.unit_is_alive(w):
                        if scenario.unit_is_alive(t):
                            state[5] = 'F'
                        else:
                            state[5] = 'T'
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
                elif t in T_target_guid:
                    index = T_target_guid.index(t)
                    weapon_index = T_best_plan.index(index)
                    for k, v in enumerate(T_number):
                        weapon_index = weapon_index - v
                        if weapon_index < 0 and T_weapon_guid[k] == w:
                            T_weapon_obe[k].unit_drop_target_contact(t)
                            break
        time = scenario.m_Duration.split('@')
        duration = int(time[0]) * 86400 + int(time[1]) * 3600 + int(time[2]) * 60
        if scenario.m_StartTime + duration <= scenario.m_Time:
            print('推演已结束！')
            sys.exit(0)
        else:
            pass


def main():
    args = parser.parse_args()
    if args.platform_mode == 'versus':
        print('比赛模式')
        ip_port = args.avail_ip_port.split(":")
        ip = ip_port[0]
        port = ip_port[1]
        # 决策步长需讨论
        env = Environment(ip, port, duration_interval=etc.DURATION_INTERVAL, app_mode=2,
                          agent_key_event_file=args.agent_key_event_file, platform_mode=args.platform_mode)
        run(env, args.side_name)
    else:
        print('开发模式')
        env = Environment(ip=etc.SERVER_IP, port=etc.SERVER_PORT, platform=etc.PLATFORM,
                          scenario_name=etc.SCENARIO_NAME, simulate_compression=etc.SIMULATE_COMPRESSION,
                          duration_interval=etc.DURATION_INTERVAL, synchronous=etc.SYNCHRONOUS, app_mode=etc.app_mode)
        run(env)


main()
if __name__ == '__main__':
    main()
