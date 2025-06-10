#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import os
import csv
import argparse
from PySide2.QtCore import QObject, Signal, Slot
import numpy as np
import timeit

from algorithm.set.radar.LEWR import Lewr
from algorithm.set.radar.TEWR import Tewr
from systemPrototype.env.env import Environment
from systemPrototype.env import etc
from algorithm.set.WTA.Functions import feasibility, probability_of_hit, get_target_A, get_weapon_set, \
    get_current_num, weapon_info, \
    get_class_num, weapon_from_target, update_weapon_SML, get_region_target, local_task, local_task2, hit_plan, \
    radar_plan
from algorithm.src.algotithmSelect import STAAlgorithmDict, WTAAlgorithmDict
from algorithm.set.WTA.Radar_plan import radarplan
# from systemPrototype.interface.src.IntellAlgTest.Dyanmic.api import *

from systemPrototype.config import current_dir

parser = argparse.ArgumentParser()
parser.add_argument("--avail_ip_port", type=str, default='127.0.0.1:6070')
parser.add_argument("--platform_mode", type=str, default='eval')
parser.add_argument("--side_name", type=str, default='红方')
parser.add_argument("--agent_key_event_file", type=str, default=None)

# file_path = r'D:\lab\WTAProgram\systemv1.0\systemPrototype\systemPrototype\interface\src\成果数据\禁忌搜索算法\resultData.csv'
# file = open(file_path, mode='a+', newline='')
# writer = csv.writer(file)

time_list = []
baseValue_list = []
targetHit_list = []


class mozi(QObject):
    """
    墨子平台python接口
    """
    # 打击方案；当前目标数；基地毁伤百分比；武器平台组合；总的目标数
    Hit_result = Signal(list, int, float, list, int, float)
    moziRunningSignal = Signal()
    TAResultAndInfo = Signal(list)
    radar1_information = Signal(list)
    radar2_information = Signal(list)
    radar3_information = Signal(list)
    radar4_information = Signal(list)
    radar5_information = Signal(list)
    radar6_information = Signal(list)

    radar1_guide_information = Signal(list)
    radar2_guide_information = Signal(list)
    radar3_guide_information = Signal(list)
    radar4_guide_information = Signal(list)
    radar5_guide_information = Signal(list)
    radar6_guide_information = Signal(list)

    Remote_Radar = Signal(list, list, list, list)
    LEWRTargetInfo = Signal(list, list, list)
    tewr_information = Signal(list, float, float, float, float, list)

    loggerDebug = Signal(str)
    loggerInfo = Signal(str)
    loggerWarning = Signal(str)

    def __init__(self):
        super(mozi, self).__init__()

    @Slot(str, str)
    def start(
            self,
            TAAlgorithmName: str,
            WTAAlgorithmName: str):
        self.args = parser.parse_args()

        if self.args.platform_mode == 'versus':
            ip_port = self.args.avail_ip_port.split(":")
            ip = ip_port[0]
            port = ip_port[1]
            # 决策步长需讨论
            env = Environment(
                ip,
                port,
                duration_interval=etc.DURATION_INTERVAL,
                app_mode=2,
                agent_key_event_file=self.args.agent_key_event_file,
                platform_mode=self.args.platform_mode)
            self.run(env, self.args.side_name)
        else:
            etc.SCENARIO_NAME = "场景ST.scen"
            env = Environment(
                etc.SERVER_IP,
                etc.SERVER_PORT,
                etc.PLATFORM,
                etc.SCENARIO_NAME,
                etc.SIMULATE_COMPRESSION,
                etc.DURATION_INTERVAL,
                etc.SYNCHRONOUS,
                etc.app_mode)

            self.run(
                env,
                TAAlgorithmName,
                WTAAlgorithmName)

    def run(
            self,
            env,
            TAAlgorithmName: str,
            WTAAlgorithmName: str,
            side_name=None):

        result_data_time = 0
        result_data_time_count = 0

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
        # print('进入推演方%s' % side_name)
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
        B1 = [facility for facility in facilities.values()
              if '基地1' in facility.strName][0]
        S1 = [facility for facility in facilities.values()
              if 'S1-1' in facility.strName][0]
        S2 = [facility for facility in facilities.values()
              if 'S1-2' in facility.strName][0]
        S3 = [facility for facility in facilities.values()
              if 'S1-3' in facility.strName][0]
        S4 = [facility for facility in facilities.values()
              if 'S1-4' in facility.strName][0]
        S5 = [facility for facility in facilities.values()
              if 'S1-5' in facility.strName][0]
        S6 = [facility for facility in facilities.values()
              if 'S1-6' in facility.strName][0]
        # 基地1的中远程武器平台
        M1 = [facility for facility in facilities.values()
              if 'M1-1' in facility.strName][0]
        M2 = [facility for facility in facilities.values()
              if 'M1-2' in facility.strName][0]
        M3 = [facility for facility in facilities.values()
              if 'M1-3' in facility.strName][0]
        M4 = [facility for facility in facilities.values()
              if 'M1-4' in facility.strName][0]
        M5 = [facility for facility in facilities.values()
              if 'M1-5' in facility.strName][0]
        # 基地1的远程武器平台
        L1 = [facility for facility in facilities.values()
              if 'L1' in facility.strName][0]
        L4 = [facility for facility in facilities.values()
              if 'L4' in facility.strName][0]
        L7 = [facility for facility in facilities.values()
              if 'L7' in facility.strName][0]
        L9 = [facility for facility in facilities.values()
              if 'L9' in facility.strName][0]
        L11 = [facility for facility in facilities.values()
               if 'L11' in facility.strName][0]
        # 基地2的近程武器平台
        B2 = [facility for facility in facilities.values()
              if '基地2' in facility.strName][0]
        S11 = [facility for facility in facilities.values()
               if 'S2-1' in facility.strName][0]
        S22 = [facility for facility in facilities.values()
               if 'S2-2' in facility.strName][0]
        S33 = [facility for facility in facilities.values()
               if 'S2-3' in facility.strName][0]
        S44 = [facility for facility in facilities.values()
               if 'S2-4' in facility.strName][0]
        S55 = [facility for facility in facilities.values()
               if 'S2-5' in facility.strName][0]
        S66 = [facility for facility in facilities.values()
               if 'S2-6' in facility.strName][0]
        # 基地2的中远程武器平台
        M11 = [facility for facility in facilities.values()
               if 'M2-1' in facility.strName][0]
        M22 = [facility for facility in facilities.values()
               if 'M2-2' in facility.strName][0]
        M33 = [facility for facility in facilities.values()
               if 'M2-3' in facility.strName][0]
        M44 = [facility for facility in facilities.values()
               if 'M2-4' in facility.strName][0]
        M55 = [facility for facility in facilities.values()
               if 'M2-5' in facility.strName][0]
        # 基地2的远程武器平台
        L2 = [facility for facility in facilities.values()
              if 'L2' in facility.strName][0]
        L5 = [facility for facility in facilities.values()
              if 'L5' in facility.strName][0]
        # 基地3的近程武器平台
        B3 = [facility for facility in facilities.values()
              if '基地3' in facility.strName][0]
        S111 = [facility for facility in facilities.values()
                if 'S3-1' in facility.strName][0]
        S222 = [facility for facility in facilities.values()
                if 'S3-2' in facility.strName][0]
        S333 = [facility for facility in facilities.values()
                if 'S3-3' in facility.strName][0]
        S444 = [facility for facility in facilities.values()
                if 'S3-4' in facility.strName][0]
        S555 = [facility for facility in facilities.values()
                if 'S3-5' in facility.strName][0]
        S666 = [facility for facility in facilities.values()
                if 'S3-6' in facility.strName][0]
        # 基地3的中远程武器平台
        M111 = [facility for facility in facilities.values()
                if 'M3-1' in facility.strName][0]
        M222 = [facility for facility in facilities.values()
                if 'M3-2' in facility.strName][0]
        M333 = [facility for facility in facilities.values()
                if 'M3-3' in facility.strName][0]
        M444 = [facility for facility in facilities.values()
                if 'M3-4' in facility.strName][0]
        M555 = [facility for facility in facilities.values()
                if 'M3-5' in facility.strName][0]
        L3 = [facility for facility in facilities.values()
              if 'L3' in facility.strName][0]
        L6 = [facility for facility in facilities.values()
              if 'L6' in facility.strName][0]
        L8 = [facility for facility in facilities.values()
              if 'L8' in facility.strName][0]
        L10 = [facility for facility in facilities.values()
               if 'L10' in facility.strName][0]
        L12 = [facility for facility in facilities.values()
               if 'L12' in facility.strName][0]
        L13 = [facility for facility in facilities.values()
               if 'L13' in facility.strName][0]
        L14 = [facility for facility in facilities.values()
               if 'L14' in facility.strName][0]
        L15 = [facility for facility in facilities.values()
               if 'L15' in facility.strName][0]
        L16 = [facility for facility in facilities.values()
               if 'L16' in facility.strName][0]
        T400 = [facility for facility in facilities.values(
        ) if 'S-400E (T1)' in facility.strName][0]
        T4001 = [facility for facility in facilities.values(
        ) if 'S-400E (T2)' in facility.strName][0]
        T4002 = [facility for facility in facilities.values(
        ) if 'S-400E (T3)' in facility.strName][0]
        T4003 = [facility for facility in facilities.values(
        ) if 'S-400E (T4)' in facility.strName][0]

        M300_1 = [facility for facility in facilities.values()
                  if 'S300-1' in facility.strName][0]
        M300_2 = [facility for facility in facilities.values()
                  if 'S300-2' in facility.strName][0]
        M300_3 = [facility for facility in facilities.values()
                  if 'S300-3' in facility.strName][0]
        M300_4 = [facility for facility in facilities.values()
                  if 'S300-4' in facility.strName][0]
        M300_5 = [facility for facility in facilities.values()
                  if 'S300-5' in facility.strName][0]
        M300_6 = [facility for facility in facilities.values()
                  if 'S300-6' in facility.strName][0]
        M300_7 = [facility for facility in facilities.values()
                  if 'S300-7' in facility.strName][0]
        M300_8 = [facility for facility in facilities.values()
                  if 'S300-8' in facility.strName][0]
        M300_9 = [facility for facility in facilities.values()
                  if 'S300-9' in facility.strName][0]

        TH1 = [facility for facility in facilities.values()
               if 'TH1' in facility.strName][0]
        TH2 = [facility for facility in facilities.values()
               if 'TH2' in facility.strName][0]
        '''
        初始化搜索雷达
        '''
        # HQ-16制导雷达
        ZD_161 = [facility for facility in facilities.values() if 'HQ-16-1' in facility.strName][0]
        ZD_162 = [facility for facility in facilities.values() if 'HQ-16-2' in facility.strName][0]
        ZD_163 = [facility for facility in facilities.values() if 'HQ-16-3' in facility.strName][0]
        ZD_9 = [facility for facility in facilities.values() if 'Radar (HQ-9-1)' in facility.strName][0]
        ZD_92 = [facility for facility in facilities.values() if 'Radar (HQ-9-2)' in facility.strName][0]
        ZD_93 = [facility for facility in facilities.values() if 'Radar (HQ-9-3)' in facility.strName][0]
        ZD_171 = [facility for facility in facilities.values() if 'Radar (HQ-17-1)' in facility.strName][0]
        ZD_172 = [facility for facility in facilities.values() if 'Radar (HQ-17-2)' in facility.strName][0]
        ZD_4001 = [facility for facility in facilities.values() if 'Radar (92N2-1)' in facility.strName][0]
        ZD_4002 = [facility for facility in facilities.values() if 'Radar (92N2-2)' in facility.strName][0]

        # 干奶酪高空搜索c
        Search_96L6E_1 = [facility for facility in facilities.values() if '“干酪板”雷达1' in facility.strName][0]
        Search_96L6E_2 = [facility for facility in facilities.values() if '“干酪板”雷达2' in facility.strName][0]
        Search_96L6E_3 = [facility for facility in facilities.values() if '“干酪板”雷达3' in facility.strName][0]

        # 远程预警雷达，用于搜索屏参数优化
        Remote_warning = [facility for facility in facilities.values() if '“铺路爪”远程预警雷达' in facility.strName][0]
        # 前置雷达和后置雷达
        Front_warning = [facility for facility in facilities.values() if 'AN/TPY-2前置' in facility.strName][0]
        After_warning_1 = [facility for facility in facilities.values() if '后置导弹防御系统1' in facility.strName][0]
        After_warning_2 = [facility for facility in facilities.values() if '后置导弹防御系统2' in facility.strName][0]
        '''
        搜索雷达与火控雷达组合
        '''
        Search_radars = {'Search_96L6E_1': Search_96L6E_1, 'Search_96L6E_2': Search_96L6E_2,
                         'Search_96L6E_3': Search_96L6E_3}
        Search_radars_combine1 = ['Search_96L6E_1', 'Search_96L6E_2', 'Search_96L6E_3']
        Search_radars_combine2 = ['Search_96L6E_1', 'Search_96L6E_2', 'Search_96L6E_3']
        Search_radars_combine3 = ['Search_96L6E_1', 'Search_96L6E_2', 'Search_96L6E_3']

        FCR_radars_combine1 = ['ZD_161', 'ZD_162', 'ZD_163']
        FCR_radars_combine2 = ['ZD_161', 'ZD_162', 'ZD_163']
        FCR_radars_combine3 = ['ZD_161', 'ZD_162', 'ZD_163']
        FCR_radars = {'ZD_161': ZD_161, 'ZD_162': ZD_162, 'ZD_163': ZD_163}

        '''
        初始化搜索雷达
        '''
        # 挂架；武器编号；射程；目标高度；目标速度；名称；导弹动力系数；基础命中率；挂架上导弹数；初始总数
        str_obe = {
            'S1': S1,
            'S2': S2,
            'S3': S3,
            'S4': S4,
            'S5': S5,
            'S6': S6,
            'M1': M1,
            'M2': M2,
            'M3': M3,
            'M4': M4,
            'M5': M5,
            'L1': L1,
            'S11': S11,
            'S22': S22,
            'S33': S33,
            'S44': S44,
            'S55': S55,
            'S66': S66,
            'M11': M11,
            'M22': M22,
            'M33': M33,
            'M44': M44,
            'M55': M55,
            'L2': L2,
            'S111': S111,
            'S222': S222,
            'S333': S333,
            'S444': S444,
            'S555': S555,
            'S666': S666,
            'M111': M111,
            'M222': M222,
            'M333': M333,
            'M444': M444,
            'M555': M555,
            'L3': L3,
            'L4': L4,
            'L5': L5,
            'L6': L6,
            'L7': L7,
            'L8': L8,
            'L9': L9,
            'L10': L10,
            'L11': L11,
            'L12': L12,
            'L13': L13,
            'L14': L14,
            'L15': L15,
            'L16': L16,
            'T400': T400,
            'T4001': T4001,
            'T4002': T4002,
            'T4003': T4003,
            'M300_1': M300_1,
            'M300_2': M300_2,
            'M300_3': M300_3,
            'M300_4': M300_4,
            'M300_5': M300_5,
            'M300_6': M300_6,
            'M300_7': M300_7,
            'M300_8': M300_8,
            'M300_9': M300_9,
            'TH1': TH1,
            'TH2': TH2}
        # 暂未更
        key_obe = {'HQ-9A（L1）': 'L1', 'HQ-9A（L2）': 'L2', 'HQ-9A（L3）': 'L3', 'HQ-16B（M1-1）': 'M1', 'HQ-16B（M1-2）': 'M2',
                   'HQ-16B（M2-1）': 'M11',
                   'HQ-16B（M2-2）': 'M22', 'HQ-16B（M3-1）': 'M111', 'HQ-16B（M3-2）': 'M222', 'HQ-17（S1-1）': 'S1',
                   'HQ-17（S1-2）': 'S2', 'HQ-17（S1-3）': 'S3',
                   'HQ-17（S2-1）': 'S11', 'HQ-17（S2-2）': 'S22', 'HQ-17（S2-3）': 'S33', 'HQ-17（S3-1）': 'S111',
                   'HQ-17（S3-2）': 'S222', 'HQ-17（S3-3）': 'S333', 'HQ-16B（S300-1）': 'M300_1', 'HQ-16B（S300-2）': 'M300_2',
                   'S-400E (SL)': 'T400'}
        radar_obe = {'M161(HQ-16ZD)': ZD_161, 'M162(HQ-16ZD)': ZD_162, 'M163(HQ-16ZD)': ZD_163,
                    'S171(HQ-17ZD)': ZD_171, 'S172(HQ-17ZD)': ZD_172,
                     'L91(HQ-9)': ZD_9, 'L92(HQ-9)': ZD_92, 'L93(HQ-9)': ZD_93,
                     'T4001(92N2)':ZD_4001, 'T4002(92N2)':ZD_4002}
        radar_range = {'M161(HQ-16ZD)': 120, 'M162(HQ-16ZD)': 120, 'M163(HQ-16ZD)': 120,
                       'S171(HQ-17ZD)': 60, 'S172(HQ-17ZD)': 60,
                       'L91(HQ-9)': 180, 'L92(HQ-9)': 180, 'L93(HQ-9)': 180,
                       'T4001(92N2)':400, 'T4002(92N2)':400}
        radar_channel = {'M161(HQ-16ZD)': 30, 'M162(HQ-16ZD)': 30, 'M163(HQ-16ZD)': 30,
                         'S171(HQ-17ZD)': 20, 'S172(HQ-17ZD)': 20,
                         'L91(HQ-9)': 40, 'L92(HQ-9)': 40, 'L93(HQ-9)': 40, 'AN/TPY-2': 20,
                         'T4001(92N2)':60, 'T4002(92N2)':60}
        fes_pair = [('S', 'S'), ('M', 'M'), ('L', 'L'), ('T', 'T')]
        weapon_radar_link = {}
        target_a = {
            '干扰机': 1,
            '预警机': 1,
            '女武神无人机': 4,
            'F-16DJ': 4.9,
            'B-1B轰炸机': 1,
            'B-52H轰炸机': 1.5,
            'RQ-180': 1,
            'F-15E战斗轰炸机': 4.5,
            '超级眼镜蛇直升机': 2,
            'F-16CM战斗机': 4.9,
            '枪骑兵轰炸机': 2,
            '巡飞弹': 1,
            '高超声速导弹': 0,
            '导弹': 0,
            '炸弹': 0}
        target_name = [
            '干扰机',
            '预警机',
            '女武神无人机',
            'F-16DJ',
            'B-1B轰炸机',
            'B-52H轰炸机',
            'RQ-180',
            'F-15E战斗轰炸机',
            '超级眼镜蛇直升机',
            'F-16CM战斗机',
            '枪骑兵轰炸机',
            '巡飞弹',
            '高超声速导弹',
            '导弹',
            '炸弹']
        special_target = ['高超声速导弹', '女武神无人机', '巡飞弹']
        special_target_L = ['超级眼镜蛇直升机', 'F-16DJ', '高超声速导弹', '女武神无人机', '巡飞弹']

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
        time_count = 190  # 5
        NET = 1
        ZERO = [[0, 0, 0, 0] for _ in range(4)]

        # 各区域yuan
        S1_original_weapon = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        S2_original_weapon = ['S11', 'S22', 'S33', 'S44', 'S55', 'S66']
        S3_original_weapon = ['S111', 'S222', 'S333', 'S444', 'S555', 'S666']
        M1_original_weapon = ['M1', 'M2', 'M3', 'M4', 'M5']
        M2_original_weapon = ['M11', 'M22', 'M33', 'M44', 'M55']
        M3_original_weapon = ['M111', 'M222', 'M333', 'M444', 'M555']
        L1_original_weapon = ['L1', 'L4', 'L7', 'L9', 'L11', 'L2', 'L5']
        L2_original_weapon = ['L3', 'L6', 'L8', 'L10', 'L12', 'L13', 'L14', 'L15', 'L16']
        # 各区域候选武器平台
        S1_prepare = ['M300_1', 'M300_3', 'M300_5', 'M300_7']
        S2_prepare = ['M300_1', 'M300_2', 'M1', 'M111']
        S3_prepare = ['M300_2', 'M300_4', 'M300_6', 'M300_7']
        M1_prepare = ['M300_3', 'M300_8', 'M300_1', 'L1', 'L4', 'L7']
        M2_prepare = ['M1', 'M111', 'M2', 'M222', 'M3', 'M333', 'L1', 'L2', 'L3', 'T400']
        M3_prepare = ['M300_4', 'M300_2', 'M300_9', 'L3', 'L6', 'L8']
        L1_prepare = ['T400', 'T4002', 'T4001']
        L2_prepare = ['T400', 'T4003', 'T4001']
        All_weapon = [
            'L1',
            'L2',
            'L3',
            'L4',
            'L5',
            'L6',
            'L7',
            'L8',
            'L9',
            'L10',
            'L11',
            'L12',
            'L13',
            'L14',
            'L15',
            'L16',
            'M1',
            'M2',
            'M3',
            'M4',
            'M5',
            'M11',
            'M22',
            'M33',
            'M44',
            'M55',
            'M111',
            'M222',
            'M333',
            'M444',
            'M555',
            'S1',
            'S2',
            'S3',
            'S4',
            'S5',
            'S6',
            'S11',
            'S22',
            'S33',
            'S44',
            'S55',
            'S66',
            'S111',
            'S222',
            'S333',
            'S444',
            'S555',
            'S666',
            'M300_1',
            'M300_2',
            'M300_3',
            'M300_4',
            'M300_5',
            'M300_6',
            'M300_7',
            'M300_8',
            'M300_9',
            'T400',
            'T4001',
            'T4002',
            'T4003',
            'TH1',
            'TH2',
            ]
        # 各区域特殊调用武器平台
        S1_special = ['T400', 'T4002', 'T4001', 'T4003']
        S2_special = ['T400', 'T4001', 'T4002', 'T4003']
        S3_special = [ 'T400', 'T4003', 'T4001', 'T4002']
        M1_special = ['T400', 'T4002', 'T4001', 'T4003']
        M2_special = ['T400', 'T4001', 'T4002', 'T4003']
        M3_special = ['T400', 'T4003', 'T4001', 'T4002']
        L1_special = ['T400', 'T4002', 'T4001', 'T4003']
        L2_special = ['T400', 'T4003', 'T4001', 'T4002']
        #
        All_weapon_obe = [
            L1,
            L2,
            L3,
            L4,
            L5,
            L6,
            L7,
            L8,
            L9,
            L10,
            L11,
            L12,
            L13,
            L14,
            L15,
            L16,
            M1,
            M2,
            M3,
            M4,
            M5,
            M11,
            M22,
            M33,
            M44,
            M55,
            M111,
            M222,
            M333,
            M444,
            M555,
            S1,
            S2,
            S3,
            S4,
            S5,
            S6,
            S11,
            S22,
            S33,
            S44,
            S55,
            S66,
            S111,
            S222,
            S333,
            S444,
            S555,
            S666,
            M300_1,
            M300_2,
            M300_3,
            M300_4,
            M300_5,
            M300_6,
            M300_7,
            M300_8,
            M300_9,
            T400,
            T4001,
            T4002,
            T4003,
            TH1,
            TH2]
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

        # 暂存发射的武器和目标
        Result_plan = [[]]
        # 每一轮的打击计划
        sub_plan = []
        # 暂不可用武器平台
        Temp_unused = []
        # 总的目标数量
        TARGET = []
        # 武器平台弹药数量
        new_mounts = every_weapon_mounts.copy()

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
        ############################
        Target_combine = []
        Exit_target = []

        everytime_information = [[]]
        radar1_init = 1
        radar2_init = 1
        radar3_init = 1
        radar4_init = 1

        radar1_ = 1
        radar2_ = 1
        radar3_ = 1
        radar4_ = 1

        radar1_pos = []
        radar2_pos = []
        radar3_pos = []
        radar4_pos = []

        Remote_count = 1
        #############################
        Weapon_combine = []
        Weapon_target_combine = []
        Exit_weapon = []
        everytime_weapon_information = []
        everytime_target_information = []

        radar1_guide_init = 1
        radar2_guide_init = 1
        radar3_guide_init = 1
        radar4_guide_init = 1

        radar1_guide_pos = []
        radar2_guide_pos = []
        radar3_guide_pos = []
        radar4_guide_pos = []

        # 控制步
        guide_step1 = 0
        guide_step2 = 0
        guide_step3 = 0
        guide_step4 = 0

        next_round1 = 0
        next_round2 = 0
        next_round3 = 0
        next_round4 = 0

        # 区域目标
        dict_zone = {}

        with open(os.path.join(current_dir, r'interface\utils\zone.csv')) as myFile:
            lines = csv.reader(myFile)
            for line in lines:
                dict_zone[line[0]] = [(float(line[1]), float(line[2])), (float(line[3]), float(line[4]))]
        while True:
            self.loggerInfo.emit("墨子线程开始")
            while time_count > 0:
                env.step()
                time_count -= 1
            count += 1
            temp_radar_channel = copy.deepcopy(radar_channel)
            for k, v in enumerate(Base_guid):
                if red_side.get_unit_by_guid(v) is None:
                    Des_base[k] = 1 * V_a[0][k]
                else:
                    Des_base[k] = 0.01 * \
                                  float(Base_obe[k].strDamageState) * V_a[0][k]

            Des = sum(Des_base) / sum(V_a[0])
            # 选择算法
            TA = STAAlgorithmDict[TAAlgorithmName]
            Al = WTAAlgorithmDict[WTAAlgorithmName]

            # 目标的对象
            # start = timeit.default_timer()
            contacts_dic = red_side.contacts
            # 目标速度大于5判定为空中目标
            targets = [item for item in contacts_dic.values()
                       if item.fCurrentSpeed > 5]
            for i in targets:
                i.set_mark_contact('H')

            if count % 1 == 0:
                targets_state = [[target.dLongitude, target.dLatitude, target.fCurrentSpeed, target.fCurrentHeading] for
                                 target in targets if
                                 '弹道导弹' not in target.strName and '核弹炸药' not in target.strName and '不明' not in target.strName and '高超声速导弹' not in target.strName]
                targets_name = [target.strName for target in targets if
                                '弹道导弹' not in target.strName and '核弹炸药' not in target.strName and '不明' not in target.strName and '高超声速导弹' not in target.strName]
                targets_dict = dict(zip(targets_name, targets_state))
                # print(targets_name)
                # self.radarInfo.emit(targets_dict)
                ###############################################
                target_information = targets_dict

                # 字典的key，即目标的名字
                target_key = list(target_information.keys())
                if not Target_combine:
                    Target_combine.append(target_key)
                    Exit_target = target_key
                else:
                    # 控制组合中目标的个数
                    new_add = list(set(target_key) - set(Exit_target))
                    # if len(new_add) >= 5 or (len(new_add) > 0 and Time_count == 0):
                    #     for l in range(len(new_add) // 5 + 1):
                    #         Target_combine.append(new_add[l * 5:(l + 1) * 5])
                    #         everytime_information.append([])
                    #     Time_count = 5
                    # else:
                    #     Time_count -= 1
                    if len(new_add) >= 5:
                        Target_combine.append(new_add[:5])
                        everytime_information.append([])
                        Exit_target = set(list(Exit_target) + new_add[:5])
                for k, v in enumerate(Target_combine):
                    everytime_information[k].append(
                        [target_information[i] if i in target_key else [0, 0, 0, 0] for i in v])
                for k, v in enumerate(everytime_information):
                    if k == 0:
                        if radar1_init == 1:
                            radar1_init = radar1_init + 1
                            radar1_pos = [[Search_radars[r].dLongitude, Search_radars[r].dLatitude]
                                          for k, r in enumerate(Search_radars_combine1)]
                        # 发送最新的目标信息
                        if v[-1] != [[0, 0, 0, 0]] * len(Target_combine[k]):
                            radar1_ += 1
                            if radar1_ <= 19:
                                self.radar1_information.emit(
                                    [radar1_pos, 0, v[-1], k, Target_combine[k], Search_radars_combine1])
                            else:
                                self.radar1_information.emit([])

                        else:
                            self.radar1_information.emit([])
                    elif k == 1:
                        if radar2_init == 1:
                            radar2_init = radar2_init + 1
                            # 目标个数
                            radar2_pos = [[Search_radars[r].dLongitude, Search_radars[r].dLatitude]
                                          for k, r in enumerate(Search_radars_combine1)]

                        if v[-1] != [[0, 0, 0, 0]] * len(Target_combine[k]):
                            radar2_ += 1
                            if radar2_ <= 19:
                                self.radar2_information.emit(
                                    [radar2_pos, 0, v[-1], k, Target_combine[k], Search_radars_combine1])
                            else:
                                self.radar2_information.emit([])

                        else:
                            self.radar2_information.emit([])
                    elif k == 2:
                        if radar3_init == 1:
                            radar3_init = radar3_init + 1
                            # 目标个数
                            radar3_pos = [[Search_radars[r].dLongitude, Search_radars[r].dLatitude]
                                          for k, r in enumerate(Search_radars_combine2)]
                        if v[-1] != [[0, 0, 0, 0]] * len(Target_combine[k]):
                            radar3_ += 1
                            if radar3_ <= 19:
                                self.radar3_information.emit(
                                    [radar3_pos, 0, v[-1], k, Target_combine[k], Search_radars_combine2])
                            else:
                                self.radar3_information.emit([])

                        else:
                            self.radar3_information.emit([])
                    elif k == 3:
                        if radar4_init == 1:
                            radar4_init = radar4_init + 1
                            # 目标个数
                            radar4_pos = [[Search_radars[r].dLongitude, Search_radars[r].dLatitude]
                                          for k, r in enumerate(Search_radars_combine2)]

                        if v[-1] != [[0, 0, 0, 0]] * len(Target_combine[k]):
                            radar4_ += 1
                            if radar4_ <= 19:
                                self.radar4_information.emit(
                                    [radar4_pos, 0, v[-1], k, Target_combine[k], Search_radars_combine2])

                            else:
                                self.radar4_information.emit([])

                        else:
                            self.radar4_information.emit([])
            now_targets = len(targets)
            targets_copy = targets.copy()
            targets_guid = [T.strGuid for T in targets]
            TARGET += targets_guid
            TARGET = list(set(TARGET))

            sum_TARGET = len(TARGET)

            # 已被分配武器的目标不作为下一阶段打击的目标
            for g in fired:
                if g in targets_guid:
                    index = targets_guid.index(g)
                    temp_t = targets_copy[index]
                    if temp_t in targets:
                        targets.remove(temp_t)
            S1_target, S2_target, S3_target, M1_target, M2_target, M3_target, L1_target, L2_target, T_target = [
            ], [], [], [], [], [], [], [], []
            Zone_target = {'S1': S1_target, 'S2': S2_target, 'S3': S3_target, 'M1': M1_target, 'M2': M2_target,
                           'M3': M3_target, 'L1': L1_target, 'L2': L2_target}
            # 进行目标的划分
            for target in targets:
                # 目标纬度
                latitude = float(target.dLatitude)
                # 目标经度
                longitude = float(target.dLongitude)
                if '弹道导弹' in target.strName or '核弹炸药' in target.strName:
                    T_target.append(target)
                else:
                    for k, v in dict_zone.items():
                        lon_min, lon_max = v[0][0], v[0][1]
                        lat_min, lat_max = v[1][0], v[1][1]
                        Flag_in = (lon_min <= longitude <= lon_max) and (lat_min <= latitude <= lat_max)
                        if k == 'ZONE':
                            if not Flag_in:
                                break
                        elif Flag_in:
                            Zone_target[k].append(target)
                            break
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
                    current = get_current_num(
                        All_weapon_obe[w].get_weapon_infos())
                    new_mounts[All_weapon[w]] = current
                    if current == 0:
                        Temp_unused.append(All_weapon[w])
            # 不可用的武器置0
            for i in Temp_unused:
                new_mounts[i] = 0
            # 原始武器更新
            M1_original_weapon_obe = [str_obe[i] for i in M1_original_weapon]
            M2_original_weapon_obe = [str_obe[i] for i in M2_original_weapon]
            M3_original_weapon_obe = [str_obe[i] for i in M3_original_weapon]
            L1_original_weapon_obe = [str_obe[i] for i in L1_original_weapon]
            L2_original_weapon_obe = [str_obe[i] for i in L2_original_weapon]

            S1_weapon, S2_weapon, S3_weapon = [], [], []
            M1_weapon, M2_weapon, M3_weapon = [], [], []
            L1_weapon, L2_weapon, T_weapon = [], [], []
            Temp_unused = []
            # 记录打击的目标
            last_target_weapon = []
            last_radar_assignment = []
            T_target_sum, Threat_T, qjk_T, T_target_v, T_target_h, T_target_name, T_target_a, T_class_num = None, [
            ], None, None, None, None, None, {}
            if targets:
                self.loggerInfo.emit("开始进行信息融合、威胁评估")
            if T_target:

                # 目标数量
                T_target_sum = len(T_target)
                # 各目标速度：list
                T_target_v = [target.fCurrentSpeed for target in T_target]
                # 各目标高度：list
                T_target_h = [
                    target.fCurrentAltitude_ASL for target in T_target]
                # 目标名字
                T_target_name = [target.strName for target in T_target]
                '''
                远程预警雷达
                '''
                if Remote_count == 1:
                    Remote_count += 1
                    Remote_VT = [v * 1.852 / 3600 for v in T_target_v]
                    Remote_distance = [float(Remote_warning.get_range_to_contact(T_guid)) * 1.852 for T_guid in
                                       T_target_guid]

                    Remote_azimuth = [target.fCurrentHeading - 90 for target in T_target]
                    Remote_elevation = [np.arctan(h / 1000 / d) / 3.14 * 180 for h, d in
                                        zip(T_target_h, Remote_distance)]
                    c_distance = [np.sqrt(x ** 2 + (y / 1000) ** 2) for x, y in zip(Remote_distance, T_target_h)]
                    Remote_beta = [np.pi / 6] * T_target_sum
                    Remote_theta_0 = [np.pi / 720] * T_target_sum
                    Remote_theta = [np.pi / 360] * T_target_sum

                    #     # 起始俯仰角
                    self.LEWRTargetInfo.emit(c_distance, Remote_azimuth, Remote_elevation)
                    azimuth_start, azimuth_end, pitch_start, pitch_end = Lewr(Remote_VT, Remote_azimuth,
                                                                              Remote_elevation,
                                                                              c_distance, Remote_beta, Remote_theta_0,
                                                                              Remote_theta)
                    # 起始方位角，终止方位角，起始俯仰角，终止俯仰角
                    self.Remote_Radar.emit(azimuth_start, azimuth_end, pitch_start, pitch_end)

                '''
                远程预警雷达
                '''
                # 目标机动系数
                T_target_a, T_target_class = get_target_A(
                    T_target_name, target_name, target_a)
                T_class_num = get_class_num(T_target_class)
                # 各目标的所有可得信息，用于威胁评估 list[[速度，高度，经度，纬度，朝向，倾斜角，翻转角，名称]
                T_target_inf = [[target.strName,
                                 target.fCurrentSpeed,
                                 target.fCurrentAltitude_ASL,
                                 target.dLongitude,
                                 target.dLatitude,
                                 target.fCurrentHeading] for target in T_target]
                Threat_T, Damage_T, Base_T, infoList_T = TA(T_target_inf).run()

                self.TAResultAndInfo.emit(infoList_T)

                # 目标打击基地的概率值
                # T_to_A = np.random.randint(0, 2, [1, T_target_sum])[0]
                qjk_T = np.zeros((T_target_sum, 3))
                for k, v in enumerate(Base_T):
                    qjk_T[k][v] = Damage_T[k]
            # 信息融合、威胁评估模块
            S1_target_sum, Threat_S1, qjk_S1, S1_target_v, S1_target_h, S1_target_name, S1_target_a, S1_class_num, S1_infoList = get_region_target(
                S1_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(S1_infoList)
            # 信息融合、威胁评估模块
            S2_target_sum, Threat_S2, qjk_S2, S2_target_v, S2_target_h, S2_target_name, S2_target_a, S2_class_num, S2_infoList = get_region_target(
                S2_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(S2_infoList)
            # 信息融合、威胁评估模块
            S3_target_sum, Threat_S3, qjk_S3, S3_target_v, S3_target_h, S3_target_name, S3_target_a, S3_class_num, S3_infoList = get_region_target(
                S3_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(S3_infoList)
            # 信息融合、威胁评估模块
            M1_target_sum, Threat_M1, qjk_M1, M1_target_v, M1_target_h, M1_target_name, M1_target_a, M1_class_num, M1_infoList = get_region_target(
                M1_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(M1_infoList)
            # 信息融合、威胁评估模块
            M2_target_sum, Threat_M2, qjk_M2, M2_target_v, M2_target_h, M2_target_name, M2_target_a, M2_class_num, M2_infoList = get_region_target(
                M2_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(M2_infoList)
            # 信息融合、威胁评估模块
            M3_target_sum, Threat_M3, qjk_M3, M3_target_v, M3_target_h, M3_target_name, M3_target_a, M3_class_num, M3_infoList = get_region_target(
                M3_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(M3_infoList)
            # 信息融合、威胁评估模块
            L1_target_sum, Threat_L1, qjk_L1, L1_target_v, L1_target_h, L1_target_name, L1_target_a, L1_class_num, L1_infoList = get_region_target(
                L1_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(L1_infoList)
            L2_target_sum, Threat_L2, qjk_L2, L2_target_v, L2_target_h, L2_target_name, L2_target_a, L2_class_num, L2_infoList = get_region_target(
                L2_target, target_name, target_a, TA, num_base=3)
            self.TAResultAndInfo.emit(L2_infoList)

            self.loggerInfo.emit("信息融合、威胁评估完成")
            # 根据各区域的威胁、数量、种类等分配合适的武器
            # 根据特殊目标分配特殊武器,并执行本地任务
            # new_mounts = every_weapon_mounts.copy()

            if count % 10 == 0:
                THREAT = [max(0.01, sum(Threat_L1)), max(0.01, sum(Threat_L2)), max(0.01, sum(Threat_M1)),
                          max(0.01, sum(Threat_M2)), max(0.01, sum(Threat_M3)),
                          max(0.01, sum(Threat_S1)), max(0.01, sum(Threat_S2)), max(0.01, sum(Threat_S3))]
                every_threat = [t / sum(THREAT) for t in THREAT]
                Radar_tewr = Tewr(len(THREAT), every_threat)
                G_best, RE0, RE1, tf0, tf1 = Radar_tewr.run()
                self.tewr_information.emit(G_best, RE0, RE1, tf0, tf1, every_threat)
            if NET == 1:
                # 本地任务
                S1_new, S1_target_norm, new_mounts = local_task(S1_target, S1_original_weapon, new_mounts, S1_class_num,
                                                                special_target, S1_special, S1_target_sum)
                S2_new, S2_target_norm, new_mounts = local_task(S2_target, S2_original_weapon, new_mounts, S2_class_num,
                                                                special_target, S2_special, S2_target_sum)
                S3_new, S3_target_norm, new_mounts = local_task(S3_target, S3_original_weapon, new_mounts, S3_class_num,
                                                                special_target, S3_special, S3_target_sum)
                # 本地任务
                M1_new, M1_target_norm, new_mounts = local_task2(M1_target, M1_class_num, special_target, M1_special,
                                                                 new_mounts, M1_original_weapon, M1_target_sum,
                                                                 M1_original_weapon_obe, str_obe)
                M2_new, M2_target_norm, new_mounts = local_task2(M2_target, M2_class_num, special_target, M2_special,
                                                                 new_mounts, M2_original_weapon, M2_target_sum,
                                                                 M2_original_weapon_obe, str_obe)
                M3_new, M3_target_norm, new_mounts = local_task2(M3_target, M3_class_num, special_target, M3_special,
                                                                 new_mounts, M3_original_weapon, M3_target_sum,
                                                                 M3_original_weapon_obe, str_obe)
                L1_new, L1_target_norm, new_mounts = local_task2(L1_target, L1_class_num, special_target_L, L1_special,
                                                                 new_mounts, L1_original_weapon, L1_target_sum,
                                                                 L1_original_weapon_obe, str_obe)
                L2_new, L2_target_norm, new_mounts = local_task2(L2_target, L2_class_num, special_target_L, L2_special,
                                                                 new_mounts, L2_original_weapon, L2_target_sum,
                                                                 L2_original_weapon_obe, str_obe)
                for i in S1_target:
                    if float(i.fCurrentAltitude_ASL) > 6000:
                        S1_target_norm += 1
                for i in S2_target:
                    if float(i.fCurrentAltitude_ASL) > 6000:
                        S2_target_norm += 1
                for i in S3_target:
                    if float(i.fCurrentAltitude_ASL) > 6000:
                        S3_target_norm += 1

                self.loggerInfo.emit("本地任务")
                '''
                更新了武器的打击end
                '''
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
                S1_weapon_obe, S1_max_r, S1_weapon_guid, S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
                    S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_sum, S1_weapon_every, S1_init_num = update_weapon_SML(
                    S1_weapon, str_obe, weapon_info)
                for k, v in S2_new.items():
                    if v > 0:
                        S2_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                S2_weapon_obe, S2_max_r, S2_weapon_guid, S2_number, S2_weapon_id, S2_weapon_name, S2_weapon_range, S2_weapon_h, \
                    S2_weapon_v, S2_rocket, S2_max_v, S2_max_l, S2_pof, S2_weapon_sum, S2_weapon_every, S2_init_num = update_weapon_SML(
                    S2_weapon, str_obe, weapon_info)
                for k, v in S3_new.items():
                    if v > 0:
                        S3_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                S3_weapon_obe, S3_max_r, S3_weapon_guid, S3_number, S3_weapon_id, S3_weapon_name, S3_weapon_range, S3_weapon_h, \
                    S3_weapon_v, S3_rocket, S3_max_v, S3_max_l, S3_pof, S3_weapon_sum, S3_weapon_every, S3_init_num = update_weapon_SML(
                    S3_weapon, str_obe, weapon_info)
                for k, v in M1_new.items():
                    if v > 0:
                        M1_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                M1_weapon_obe, M1_max_r, M1_weapon_guid, M1_number, M1_weapon_id, M1_weapon_name, M1_weapon_range, M1_weapon_h, \
                    M1_weapon_v, M1_rocket, M1_max_v, M1_max_l, M1_pof, M1_weapon_sum, M1_weapon_every, M1_init_num = update_weapon_SML(
                    M1_weapon, str_obe, weapon_info)
                for k, v in M2_new.items():
                    if v > 0:
                        M2_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                M2_weapon_obe, M2_max_r, M2_weapon_guid, M2_number, M2_weapon_id, M2_weapon_name, M2_weapon_range, M2_weapon_h, \
                    M2_weapon_v, M2_rocket, M2_max_v, M2_max_l, M2_pof, M2_weapon_sum, M2_weapon_every, M2_init_num = update_weapon_SML(
                    M2_weapon, str_obe, weapon_info)
                for k, v in M3_new.items():
                    if v > 0:
                        M3_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                M3_weapon_obe, M3_max_r, M3_weapon_guid, M3_number, M3_weapon_id, M3_weapon_name, M3_weapon_range, M3_weapon_h, \
                    M3_weapon_v, M3_rocket, M3_max_v, M3_max_l, M3_pof, M3_weapon_sum, M3_weapon_every, M3_init_num = update_weapon_SML(
                    M3_weapon, str_obe, weapon_info)
                for k, v in L1_new.items():
                    if v > 0:
                        L1_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                L1_weapon_obe, L1_max_r, L1_weapon_guid, L1_number, L1_weapon_id, L1_weapon_name, L1_weapon_range, L1_weapon_h, \
                    L1_weapon_v, L1_rocket, L1_max_v, L1_max_l, L1_pof, L1_weapon_sum, L1_weapon_every, L1_init_num = update_weapon_SML(
                    L1_weapon, str_obe, weapon_info)
                for k, v in L2_new.items():
                    if v > 0:
                        L2_weapon.append(k)
                        weapon_info[k]['num_weapon'] = v
                L2_weapon_obe, L2_max_r, L2_weapon_guid, L2_number, L2_weapon_id, L2_weapon_name, L2_weapon_range, L2_weapon_h, \
                    L2_weapon_v, L2_rocket, L2_max_v, L2_max_l, L2_pof, L2_weapon_sum, L2_weapon_every, L2_init_num = update_weapon_SML(
                    L2_weapon, str_obe, weapon_info)
            else:
                S1_new = {}
                S1_weapon_old = ['S1', 'S2', 'S3', 'M300_1']
                for i in S1_weapon_old:
                    if new_mounts[i] > 0:
                        S1_new[i] = new_mounts[i]
                        S1_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                S1_weapon_obe, S1_max_r, S1_weapon_guid, S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
                    S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_sum, S1_weapon_every, S1_init_num = update_weapon_SML(
                    S1_weapon, str_obe, weapon_info)
                S2_new = {}
                S2_weapon_old = ['S11', 'S22', 'S33']
                for i in S2_weapon_old:
                    if new_mounts[i] > 0:
                        S2_new[i] = new_mounts[i]
                        S2_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                S2_weapon_obe, S2_max_r, S2_weapon_guid, S2_number, S2_weapon_id, S2_weapon_name, S2_weapon_range, S2_weapon_h, \
                    S2_weapon_v, S2_rocket, S2_max_v, S2_max_l, S2_pof, S2_weapon_sum, S2_weapon_every, S2_init_num = update_weapon_SML(
                    S2_weapon, str_obe, weapon_info)
                S3_new = {}
                S3_weapon_old = ['S111', 'S222', 'S333', 'M300_2']
                for i in S3_weapon_old:
                    if new_mounts[i] > 0:
                        S3_new[i] = new_mounts[i]
                        S3_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                S3_weapon_obe, S3_max_r, S3_weapon_guid, S3_number, S3_weapon_id, S3_weapon_name, S3_weapon_range, S3_weapon_h, \
                    S3_weapon_v, S3_rocket, S3_max_v, S3_max_l, S3_pof, S3_weapon_sum, S3_weapon_every, S3_init_num = update_weapon_SML(
                    S3_weapon, str_obe, weapon_info)
                M1_new = {}
                M1_weapon_old = ['M1', 'M2']
                for i in M1_weapon_old:
                    if new_mounts[i] > 0:
                        M1_new[i] = new_mounts[i]
                        M1_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                M1_weapon_obe, M1_max_r, M1_weapon_guid, M1_number, M1_weapon_id, M1_weapon_name, M1_weapon_range, M1_weapon_h, \
                    M1_weapon_v, M1_rocket, M1_max_v, M1_max_l, M1_pof, M1_weapon_sum, M1_weapon_every, M1_init_num = update_weapon_SML(
                    M1_weapon, str_obe, weapon_info)
                M2_new = {}
                M2_weapon_old = ['M11', 'M22']
                for i in M2_weapon_old:
                    if new_mounts[i] > 0:
                        M2_new[i] = new_mounts[i]
                        M2_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                M2_weapon_obe, M2_max_r, M2_weapon_guid, M2_number, M2_weapon_id, M2_weapon_name, M2_weapon_range, M2_weapon_h, \
                    M2_weapon_v, M2_rocket, M2_max_v, M2_max_l, M2_pof, M2_weapon_sum, M2_weapon_every, M2_init_num = update_weapon_SML(
                    M2_weapon, str_obe, weapon_info)
                M3_new = {}
                M3_weapon_old = ['M111', 'M222']
                for i in M3_weapon_old:
                    if new_mounts[i] > 0:
                        M3_new[i] = new_mounts[i]
                        M3_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                M3_weapon_obe, M3_max_r, M3_weapon_guid, M3_number, M3_weapon_id, M3_weapon_name, M3_weapon_range, M3_weapon_h, \
                    M3_weapon_v, M3_rocket, M3_max_v, M3_max_l, M3_pof, M3_weapon_sum, M3_weapon_every, M3_init_num = update_weapon_SML(
                    M3_weapon, str_obe, weapon_info)
                L1_new = {}
                L1_weapon_old = ['L1', 'T400']
                for i in L1_weapon_old:
                    if new_mounts[i] > 0:
                        L1_new[i] = new_mounts[i]
                        L1_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                L1_weapon_obe, L1_max_r, L1_weapon_guid, L1_number, L1_weapon_id, L1_weapon_name, L1_weapon_range, L1_weapon_h, \
                    L1_weapon_v, L1_rocket, L1_max_v, L1_max_l, L1_pof, L1_weapon_sum, L1_weapon_every, L1_init_num = update_weapon_SML(
                    L1_weapon, str_obe, weapon_info)
                L2_new = {}
                L2_weapon_old = ['L2', 'L3']
                for i in L2_weapon_old:
                    if new_mounts[i] > 0:
                        L2_new[i] = new_mounts[i]
                        L2_weapon.append(i)
                        weapon_info[i]['num_weapon'] = new_mounts[i]
                L2_weapon_obe, L2_max_r, L2_weapon_guid, L2_number, L2_weapon_id, L2_weapon_name, L2_weapon_range, L2_weapon_h, \
                    L2_weapon_v, L2_rocket, L2_max_v, L2_max_l, L2_pof, L2_weapon_sum, L2_weapon_every, L2_init_num = update_weapon_SML(
                    L2_weapon, str_obe, weapon_info)

            T_new = {}
            T_weapon_old = ['TH1', 'TH2']
            for i in T_weapon_old:
                if new_mounts[i] > 0:
                    T_new[i] = new_mounts[i]
                    T_weapon.append(i)
                    weapon_info[i]['num_weapon'] = new_mounts[i]
            T_weapon_obe, T_max_r, T_weapon_guid, T_number, T_weapon_id, T_weapon_name, T_weapon_range, T_weapon_h, \
                T_weapon_v, T_rocket, T_max_v, T_max_l, T_pof, T_weapon_sum, T_weapon_every, T_init_num = update_weapon_SML(
                T_weapon, str_obe, weapon_info)
            # 显示各空域的本地资源和调度资源
            S1_weapon_num = [[], []]
            for i in S1_weapon:
                num = S1_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in S1_original_weapon:
                    S1_weapon_num[0].append(str_weapon)
                else:
                    S1_weapon_num[1].append(str_weapon)
            S2_weapon_num = [[], []]
            for i in S2_weapon:
                num = S2_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in S2_original_weapon:
                    S2_weapon_num[0].append(str_weapon)
                else:
                    S2_weapon_num[1].append(str_weapon)
            S3_weapon_num = [[], []]
            for i in S3_weapon:
                num = S3_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in S3_original_weapon:
                    S3_weapon_num[0].append(str_weapon)
                else:
                    S3_weapon_num[1].append(str_weapon)
            M1_weapon_num = [[], []]
            for i in M1_weapon:
                num = M1_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in M1_original_weapon:
                    M1_weapon_num[0].append(str_weapon)
                else:
                    M1_weapon_num[1].append(str_weapon)
            M2_weapon_num = [[], []]
            for i in M2_weapon:
                num = M2_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in M2_original_weapon:
                    M2_weapon_num[0].append(str_weapon)
                else:
                    M2_weapon_num[1].append(str_weapon)
            M3_weapon_num = [[], []]
            for i in M3_weapon:
                num = M3_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in M3_original_weapon:
                    M3_weapon_num[0].append(str_weapon)
                else:
                    M3_weapon_num[1].append(str_weapon)
            L1_weapon_num = [[], []]
            for i in L1_weapon:
                num = L1_new[i]
                str_weapon = weapon_info[i]["name"] + "  ×  " + str(num)
                if i in L1_original_weapon:
                    L1_weapon_num[0].append(str_weapon)
                else:
                    L1_weapon_num[1].append(str_weapon)
            L2_weapon_num = [[], []]
            for i in L2_weapon:
                num = L2_new[i]
                str_weapon = weapon_info[i]["name"] + "  × " + str(num)
                if i in L2_original_weapon:
                    L2_weapon_num[0].append(str_weapon)
                else:
                    L2_weapon_num[1].append(str_weapon)
            if T_target and T_weapon:
                start = timeit.default_timer()

                T_best_plan, last_target_weapon = hit_plan(T_weapon, T_weapon_obe, T_target, T_number,
                                                           T_weapon_range,
                                                           T_weapon_v, T_target_v, T_weapon_h, T_target_h,
                                                           T_target_name, T_new, T_rocket, T_max_v, T_max_l,
                                                           T_pof,
                                                           T_target_a, T_weapon_sum, Threat_T, qjk_T, V_a,
                                                           T_target_guid, T_weapon_id, last_target_weapon, Al)
                for v in T_best_plan:
                    if v >= 0:
                        last_radar_assignment.append('AN/TPY-2')  #
                end = timeit.default_timer()
                T_time_list.append(end - start)
                T_count += 1
                result_data_time += end - start
                result_data_time_count += 1
            self.loggerInfo.emit("开始进行武器目标分配")
            # 武器目标分配模块
            if (S1_target and S1_weapon) and count % 2 == 0 or '高超声速导弹' in S1_class_num:
                start = timeit.default_timer()
                S1_best_plan, last_target_weapon = hit_plan(S1_weapon, S1_weapon_obe, S1_target, S1_number,
                                                            S1_weapon_range,
                                                            S1_weapon_v, S1_target_v, S1_weapon_h, S1_target_h,
                                                            S1_target_name, S1_new, S1_rocket, S1_max_v, S1_max_l,
                                                            S1_pof,
                                                            S1_target_a, S1_weapon_sum, Threat_S1, qjk_S1, V_a,
                                                            S1_target_guid, S1_weapon_id, last_target_weapon, Al)
                S1_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(S1_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, S1_weapon_obe,
                                                                                      S1_target,
                                                                                      S1_best_plan, Threat_S1,
                                                                                      S1_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('S1第{}轮运行时间{}'.format(S1_count, end - start))
                S1_time_list.append(end - start)
                S1_count += 1
                result_data_time += end - start
                result_data_time_count += 1
            # 武器目标分配模块
            if S2_target and S2_weapon and count % 2 == 0 or '高超声速导弹' in S2_class_num:
                start = timeit.default_timer()
                S2_best_plan, last_target_weapon = hit_plan(S2_weapon, S2_weapon_obe, S2_target, S2_number,
                                                            S2_weapon_range,
                                                            S2_weapon_v, S2_target_v, S2_weapon_h, S2_target_h,
                                                            S2_target_name, S2_new, S2_rocket, S2_max_v, S2_max_l,
                                                            S2_pof,
                                                            S2_target_a, S2_weapon_sum, Threat_S2, qjk_S2, V_a,
                                                            S2_target_guid, S2_weapon_id, last_target_weapon, Al)
                S2_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(S2_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, S2_weapon_obe,
                                                                                      S2_target,
                                                                                      S2_best_plan, Threat_S2,
                                                                                      S2_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('S2第{}轮运行时间{}'.format(S2_count, end - start))
                S2_time_list.append(end - start)
                S2_count += 1

                result_data_time += end - start
                result_data_time_count += 1
            # 武器目标分配模块
            if S3_target and S3_weapon and count % 2 == 0 or '高超声速导弹' in S3_class_num:
                start = timeit.default_timer()
                S3_best_plan, last_target_weapon = hit_plan(S3_weapon, S3_weapon_obe, S3_target, S3_number,
                                                            S3_weapon_range,
                                                            S3_weapon_v, S3_target_v, S3_weapon_h, S3_target_h,
                                                            S3_target_name, S3_new, S3_rocket, S3_max_v, S3_max_l,
                                                            S3_pof,
                                                            S3_target_a, S3_weapon_sum, Threat_S3, qjk_S3, V_a,
                                                            S3_target_guid, S3_weapon_id, last_target_weapon, Al)
                S3_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(S3_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, S3_weapon_obe,
                                                                                      S3_target,
                                                                                      S3_best_plan, Threat_S3,
                                                                                      S3_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('S3第{}轮运行时间{}'.format(S3_count, end - start))
                S3_time_list.append(end - start)
                S3_count += 1

                result_data_time += end - start
                result_data_time_count += 1

            if M1_target and M1_weapon and count % 5 == 0 or '高超声速导弹' in M1_class_num:
                start = timeit.default_timer()
                M1_best_plan, last_target_weapon = hit_plan(M1_weapon, M1_weapon_obe, M1_target, M1_number,
                                                            M1_weapon_range,
                                                            M1_weapon_v, M1_target_v, M1_weapon_h, M1_target_h,
                                                            M1_target_name, M1_new, M1_rocket, M1_max_v, M1_max_l,
                                                            M1_pof,
                                                            M1_target_a, M1_weapon_sum, Threat_M1, qjk_M1, V_a,
                                                            M1_target_guid, M1_weapon_id, last_target_weapon, Al)
                M1_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(M1_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, M1_weapon_obe,
                                                                                      M1_target,
                                                                                      M1_best_plan, Threat_M1,
                                                                                      M1_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('M1第{}轮运行时间{}'.format(M1_count, end - start))
                M1_time_list.append(end - start)
                M1_count += 1

                result_data_time += end - start
                result_data_time_count += 1
            # 武器目标分配模块
            if M2_target and M2_weapon and count % 6 == 0 or '高超声速导弹' in M2_class_num:
                start = timeit.default_timer()
                M2_best_plan, last_target_weapon = hit_plan(M2_weapon, M2_weapon_obe, M2_target, M2_number,
                                                            M2_weapon_range,
                                                            M2_weapon_v, M2_target_v, M2_weapon_h, M2_target_h,
                                                            M2_target_name, M2_new, M2_rocket, M2_max_v, M2_max_l,
                                                            M2_pof,
                                                            M2_target_a, M2_weapon_sum, Threat_M2, qjk_M2, V_a,
                                                            M2_target_guid, M2_weapon_id, last_target_weapon, Al)
                M2_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(M2_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, M2_weapon_obe,
                                                                                      M2_target,
                                                                                      M2_best_plan, Threat_M2,
                                                                                      M2_number, last_radar_assignment,
                                                                                      fes_pair)

                end = timeit.default_timer()
                # # print('M2第{}轮运行时间{}'.format(M2_count, end - start))
                M2_time_list.append(end - start)
                M2_count += 1

                result_data_time += end - start
                result_data_time_count += 1
            # 武器目标分配模块
            if M3_target and M3_weapon and count % 5 == 0 or '高超声速导弹' in M3_class_num:
                start = timeit.default_timer()
                M3_best_plan, last_target_weapon = hit_plan(M3_weapon, M3_weapon_obe, M3_target, M3_number,
                                                            M3_weapon_range,
                                                            M3_weapon_v, M3_target_v, M3_weapon_h, M3_target_h,
                                                            M3_target_name, M3_new, M3_rocket, M3_max_v, M3_max_l,
                                                            M3_pof,
                                                            M3_target_a, M3_weapon_sum, Threat_M3, qjk_M3, V_a,
                                                            M3_target_guid, M3_weapon_id, last_target_weapon, Al)
                M3_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(M3_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, M3_weapon_obe,
                                                                                      M3_target,
                                                                                      M3_best_plan, Threat_M3,
                                                                                      M3_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('M3第{}轮运行时间{}'.format(M3_count, end - start))
                M3_time_list.append(end - start)
                M3_count += 1

                result_data_time += end - start
                result_data_time_count += 1
            # 武器目标分配模块
            if L1_target and L1_weapon and count % 11 == 0 or '高超声速导弹' in L1_class_num:
                start = timeit.default_timer()
                L1_best_plan, last_target_weapon = hit_plan(L1_weapon, L1_weapon_obe, L1_target, L1_number,
                                                            L1_weapon_range,
                                                            L1_weapon_v, L1_target_v, L1_weapon_h, L1_target_h,
                                                            L1_target_name, L1_new, L1_rocket, L1_max_v, L1_max_l,
                                                            L1_pof,
                                                            L1_target_a, L1_weapon_sum, Threat_L1, qjk_L1, V_a,
                                                            L1_target_guid, L1_weapon_id, last_target_weapon, Al)
                L1_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(L1_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, L1_weapon_obe,
                                                                                      L1_target,
                                                                                      L1_best_plan, Threat_L1,
                                                                                      L1_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('L1第{}轮运行时间{}'.format(L1_count, end - start))
                L1_time_list.append(end - start)
                L1_count += 1

                result_data_time += end - start
                result_data_time_count += 1
            if L2_target and L2_weapon and count % 11 == 0 or '高超声速导弹' in L2_class_num:
                start = timeit.default_timer()
                L2_best_plan, last_target_weapon = hit_plan(L2_weapon, L2_weapon_obe, L2_target, L2_number,
                                                            L2_weapon_range,
                                                            L2_weapon_v, L2_target_v, L2_weapon_h, L2_target_h,
                                                            L2_target_name, L2_new, L2_rocket, L2_max_v, L2_max_l,
                                                            L2_pof,
                                                            L2_target_a, L2_weapon_sum, Threat_L2, qjk_L2, V_a,
                                                            L2_target_guid, L2_weapon_id, last_target_weapon, Al)
                L2_radar_plan, temp_radar_channel, last_radar_assignment = radar_plan(L2_weapon, radar_obe,
                                                                                      temp_radar_channel, radar_range,
                                                                                      radarplan, L2_weapon_obe,
                                                                                      L2_target,
                                                                                      L2_best_plan, Threat_L2,
                                                                                      L2_number, last_radar_assignment,
                                                                                      fes_pair)
                end = timeit.default_timer()
                # # print('L2第{}轮运行时间{}'.format(L2_count, end - start))
                L2_time_list.append(end - start)
                L2_count += 1

                result_data_time += end - start
                result_data_time_count += 1
            self.loggerInfo.emit("武器目标分配模块结束")
            # 武器组合
            Weapon_now = [
                S1_weapon_num,
                S2_weapon_num,
                S3_weapon_num,
                M1_weapon_num,
                M2_weapon_num,
                M3_weapon_num,
                L1_weapon_num,
                L2_weapon_num]
            # 战场更新及阶段统计
            # scenario.mozi_server.run_grpc_simulate()
            self.loggerInfo.emit("武器目标分配模块结束11111111111111111")
            scenario = env.step()
            self.loggerInfo.emit("武器目标分配模块结束22222222222222222")
            # 发射的武器信息
            sam_set = red_side.get_weapons()
            fired = [one.get_summary_info()["target"]
                     for one in sam_set.values()]
            # 筛选出HQ-16B武器，用于跟踪识别
            loggerHQ16Str = ""
            weapon_information = {}
            target_information = {}
            weapon_key = []
            weapon_target_dict = {}
            if sam_set:
                HQ16_fired = [one for one in sam_set.values() if 'HQ-16B' in one.get_summary_info()["name"]]
                HQ16_target_guid = []
                No_target_weapon = []
                # HQ16_target_guid = [one.get_summary_info()["target"] for one in HQ16_fired]
                if HQ16_fired:
                    for k, one in enumerate(HQ16_fired):
                        if one.get_summary_info()["target"]:
                            HQ16_target_guid.append(one.get_summary_info()["target"])
                        else:
                            No_target_weapon.append(one)
                    if No_target_weapon:
                        for nn in No_target_weapon:
                            HQ16_fired.remove(nn)
                # 将loggerHQ16从list转为str传给self.loggerInfo
                HQ16_target = []
                No_target_obe = []
                for k, guid in enumerate(HQ16_target_guid):
                    Flag_target = False
                    for t in targets_copy:
                        if t.strGuid == guid:
                            Flag_target = True
                            HQ16_target.append(t)
                            break
                    if not Flag_target:
                        No_target_obe.append(HQ16_fired[k])
                if No_target_obe:
                    for ww in No_target_obe:
                        HQ16_fired.remove(ww)
                if count % 1 == 0 and HQ16_fired and len(HQ16_fired) == len(HQ16_target):
                    weapons_state = [[weapon.dLongitude, weapon.dLatitude, weapon.fCurrentSpeed, weapon.fCurrentHeading]
                                     for
                                     weapon in HQ16_fired]
                    targets_state = [[target.dLongitude, target.dLatitude, target.fCurrentSpeed, target.fCurrentHeading]
                                     for
                                     target in HQ16_target]
                    weapons_name = [weapon.strName for weapon in HQ16_fired]
                    targets_name = [target.strName for target in HQ16_target]
                    nameStr = " "
                    for nameLogger in targets_name:
                        nameStr += nameLogger
                    self.loggerInfo.emit("筛选出以下目标，用于跟踪识别:" + nameStr)

                    weapons_dict = dict(zip(weapons_name, weapons_state))
                    targets_dict = dict(zip(targets_name, targets_state))
                    weapon_target_dict = dict(zip(weapons_name, targets_name))

                    ###############################################
                    weapon_information = weapons_dict
                    target_information = targets_dict

                    # 字典的key，即武器的名字
                    weapon_key = list(weapon_information.keys())

                    # 控制组合中目标的个数
                    new_add = list(set(weapon_key) - set(Exit_weapon))
                    length_add = len(new_add)
                    step_add = 0
                    while length_add >= 4:
                        Weapon_combine.append(new_add[step_add:step_add + 4])
                        Weapon_target_combine.append([weapon_target_dict[i] for i in new_add[step_add:step_add + 4]])
                        everytime_weapon_information.append([])
                        everytime_target_information.append([])
                        Exit_weapon = set(list(Exit_weapon) + new_add[step_add:step_add + 4])
                        step_add += 4
                        length_add -= 4


            for k, v in enumerate(Weapon_combine):

                wea_inf = [weapon_information[i] if i in weapon_key else [0, 0, 0, 0] for i in v]
                everytime_weapon_information[k].append(wea_inf)
                tar_inf = []
                for i in v:
                    if i in weapon_key:
                        if i in weapon_target_dict:
                            temp_target = weapon_target_dict[i]
                            if temp_target in target_information:
                                tar_inf1 = target_information[temp_target]
                            else:
                                tar_inf1 = [0,0,0,0]
                        else:
                            tar_inf1 = [0,0,0,0]
                    else:
                        tar_inf1 = [0,0,0,0]
                    tar_inf.append(tar_inf1)

                # tar_inf = [target_information[weapon_target_dict[i]] if i in weapon_key else [0, 0, 0, 0] for i in v]
                everytime_target_information[k].append(tar_inf)
            self.loggerInfo.emit("7777777777777777777777777777777")
            for k, v in enumerate(Weapon_combine):
                if k == 0:  # [radar1_pos, 0, v[-1], k, Target_combine[k], Search_radars_combine1]
                    weapon_target_combine1 = Weapon_target_combine[k + next_round1]
                    if radar1_guide_init == 1:
                        radar1_guide_init = radar1_guide_init + 1
                        radar1_guide_pos = [[FCR_radars[r].dLongitude, FCR_radars[r].dLatitude]
                                            for k, r in enumerate(FCR_radars_combine1)]
                    # 发送最新的目标信息
                    if (len(everytime_weapon_information[k + next_round1]) -1 >= guide_step1
                            and everytime_weapon_information[k + next_round1][guide_step1] != ZERO):
                        self.radar1_guide_information.emit(
                            [radar1_guide_pos, k, everytime_weapon_information[k + next_round1][guide_step1],
                             everytime_target_information[k + next_round1][guide_step1],
                             Weapon_combine[k + next_round1], weapon_target_combine1, FCR_radars_combine1])

                        guide_step1 += 1
                        self.loggerInfo.emit("制导信号1已发送")
                    else:
                        if len(everytime_weapon_information) > k + next_round1 + 4:
                            self.radar1_guide_information.emit(
                                [radar1_guide_pos, k,
                                 ZERO,
                                 ZERO,
                                 Weapon_combine[k + next_round1], weapon_target_combine1, FCR_radars_combine1])
                            next_round1 += 4
                            radar1_guide_init = 1
                            guide_step1 = 0
                        else:
                            self.radar1_guide_information.emit([])

                elif k == 1:
                    weapon_target_combine2 = Weapon_target_combine[k + next_round2]
                    if radar2_guide_init == 1:
                        radar2_guide_init = radar2_guide_init + 1
                        radar2_guide_pos = [[FCR_radars[r].dLongitude, FCR_radars[r].dLatitude]
                                            for k, r in enumerate(FCR_radars_combine2)]
                    if len(everytime_weapon_information[k + next_round2]) - 1 >= guide_step2 and everytime_weapon_information[k + next_round2][guide_step2] != ZERO:
                        self.radar2_guide_information.emit(
                            [radar2_guide_pos, k, everytime_weapon_information[k + next_round2][guide_step2],
                             everytime_target_information[k + next_round2][guide_step2],
                             Weapon_combine[k + next_round2], weapon_target_combine2, FCR_radars_combine2])
                        guide_step2 += 1
                        self.loggerInfo.emit("制导信号2已发送")
                    else:
                        if len(everytime_weapon_information) > k + next_round2 + 4:
                            self.radar2_guide_information.emit(
                                [radar2_guide_pos, k,
                                 ZERO,
                                 ZERO,
                                 Weapon_combine[k + next_round2], weapon_target_combine2, FCR_radars_combine2])
                            next_round2 += 4
                            radar2_guide_init = 1
                            guide_step2 = 0
                        else:
                            self.radar2_guide_information.emit([])

                elif k == 2:
                    weapon_target_combine3 = Weapon_target_combine[k + next_round3]
                    if radar3_guide_init == 1:
                        radar3_guide_init = radar3_guide_init + 1
                        radar3_guide_pos = [[FCR_radars[r].dLongitude, FCR_radars[r].dLatitude]
                                            for k, r in enumerate(FCR_radars_combine3)]
                    if len(everytime_weapon_information[k + next_round3]) - 1 >= guide_step3 and everytime_weapon_information[k + next_round3][guide_step3] != ZERO:
                        self.radar3_guide_information.emit(
                            [radar3_guide_pos, k,
                             everytime_weapon_information[k + next_round3][guide_step3],
                             everytime_target_information[k + next_round3][guide_step3],
                             Weapon_combine[k + next_round3], weapon_target_combine3, FCR_radars_combine3])
                        guide_step3 += 1
                        self.loggerInfo.emit("制导信号3已发送")
                    else:
                        if len(everytime_weapon_information) > k + next_round3 + 4:
                            self.radar3_guide_information.emit(
                                [radar3_guide_pos, k,
                                 ZERO,
                                 ZERO,
                                 Weapon_combine[k + next_round3], weapon_target_combine3, FCR_radars_combine3])
                            next_round3 += 4
                            radar3_guide_init = 1
                            guide_step3 = 0
                        else:
                            self.radar3_guide_information.emit([])

                elif k == 3:
                    weapon_target_combine4 = Weapon_target_combine[k + next_round4]
                    if radar4_guide_init == 1:
                        radar4_guide_init = radar4_guide_init + 1
                        radar4_guide_pos = [[FCR_radars[r].dLongitude, FCR_radars[r].dLatitude]
                                            for k, r in enumerate(FCR_radars_combine1)]
                    if len(everytime_weapon_information[k + next_round4]) - 1 >= guide_step4 and everytime_weapon_information[k + next_round4][guide_step4] != ZERO:
                        self.radar4_guide_information.emit(
                            [radar4_guide_pos, k,
                             everytime_weapon_information[k + next_round4][guide_step4],
                             everytime_target_information[k + next_round4][guide_step4],
                             Weapon_combine[k + next_round4], weapon_target_combine4, FCR_radars_combine3])
                        guide_step4 += 1
                        self.loggerInfo.emit("制导信号4已发送")
                    else:
                        if len(everytime_weapon_information) > k + next_round4 + 4:
                            self.radar4_guide_information.emit(
                                [radar4_guide_pos, k,
                                 ZERO,
                                 ZERO,
                                 Weapon_combine[k + next_round4], weapon_target_combine4, FCR_radars_combine3])
                            next_round4 += 4
                            radar4_guide_init = 1
                            guide_step4 = 0
                        else:
                            self.radar4_guide_information.emit([])
            self.loggerInfo.emit("筛选出以下HQ-16B武器，用于跟踪识别:" + loggerHQ16Str)

            loggerTargetStr = ""
            # 发射武器的各种信息
            # target:guid plat:guid weapon:name weapon:guid
            target_weapon = [
                (one.get_summary_info()["target"],
                 one.get_summary_info()["shooter"],
                 one.strName,
                 one.strGuid) for one in sam_set.values()]

            # 输出打击信息结果等
            # last_copy = copy.deepcopy(last_target_weapon)

            # 中科院CAS动态基地剩余价值率和目标拦截率预测模块
            kill_chains_CAS = {}
            weapons_info_dict_CAS = {}
            zd_info_dict_CAS = {}

            for i in target_weapon:
                # i[0]:目标guid i[1]:武器发射平台 i[2]:武器名字 i[3]:武器guid
                for k, j in enumerate(last_target_weapon):
                    # j[0]:计划中目标 j[1]:武器发射平台
                    if i[0] == j[0].strGuid and i[1] == j[1].strGuid:
                        zd = last_radar_assignment[k]
                        # print('武器平台{}发射了导弹，ZD为{}'.format(j[1].strName, zd))
                        # self.loggerInfo.emit('武器平台{}发射了导弹，ZD为{}'.format(j[1].strName, zd))
                        if zd != '666':
                            weapon_radar_link[i[3]] = zd
                            radar_channel[zd] -= 1
                        # if zd != '666' and zd != 'AN/SPY-1D(V)':
                        #     # 以dict格式构建杀伤链，信息格式为：（键）目标名称，（值）[武器名称，制导平台名称，目标名称]
                        #     kill_chains_CAS[j[0].strName] = [j[1].strName, zd, j[0].strName]
                        #     if j[1].strName in key_obe and key_obe[j[1].strName] in weapon_info:
                        #         weapons_info_dict_CAS[j[1].strName] = [j[1].dLongitude, j[0].dLatitude,
                        #                                            weapon_info[key_obe[j[1].strName]]['hit_range'][1]]
                        #
                        #     else:
                        #         weapons_info_dict_CAS[j[1].strName] = [j[1].dLongitude, j[0].dLatitude, 10]
                        #
                        #     zd_info_dict_CAS[zd] = [radar_obe[zd].dLongitude, radar_obe[zd].dLatitude, radar_range[zd]]
                        #
                        #
                        # loggerTargetStr += j[0].strName
                        sub_plan.append(
                            [j[1].strName, j[0].strName, i[2], i[3], i[0], 'x', zd])

                        last_target_weapon.remove(j)
                        last_radar_assignment.pop(k)
                        break

            # print("原始信息")
            # print(kill_chains_CAS, weapons_info_dict_CAS, zd_info_dict_CAS)
            #
            #
            # sim = Simulator()
            # sim.restart(enemy_num=len(kill_chains_CAS), enemy_target_num=3)
            #
            # red_nodes = {}
            # red_edges = {}
            # for zd_name_CAS, zd_info_CAS in zd_info_dict_CAS.items():
            #
            #     tracking = TrackingUnitData(name=zd_name_CAS, pos=np.array(zd_info_CAS[:2]), combat_enemy_radius=zd_info_CAS[-1])
            #     red_nodes[zd_name_CAS] = tracking
            #     print(zd_name_CAS, tracking)
            # for weapon_name_CAS, weapon_info_CAS in weapons_info_dict_CAS.items():
            #     firing = FiringUnitData(name=weapon_name_CAS, pos=np.array(weapon_info_CAS[:2]), combat_enemy_radius=weapon_info_CAS[-1])
            #     red_nodes[weapon_name_CAS] = firing
            #     print(weapon_name_CAS, firing)
            #
            # for zd_name_CAS, zd_info_CAS in zd_info_dict_CAS.items():
            #     for weapon_name_CAS, weapon_info_CAS in weapons_info_dict_CAS.items():
            #         red_edges[(zd_name_CAS, weapon_name_CAS)] = [red_nodes[zd_name_CAS], red_nodes[weapon_name_CAS]]
            #         print(str(zd_name_CAS)[0], str(weapon_name_CAS)[0])
            #         if (str(zd_name_CAS)[0], str(weapon_name_CAS)[0]) in fes_pair:
            #             red_edges[(zd_name_CAS, weapon_name_CAS)] = [red_nodes[zd_name_CAS], red_nodes[weapon_name_CAS]]
            # print('CAS动态基地剩余价值率和目标拦截率预测模块输入：')
            # print(len(kill_chains_CAS), len(red_nodes), len(red_edges))
            # print(len(kill_chains_CAS) > 0 and len(red_nodes) > 0 and len(red_edges) > 0)
            # if len(kill_chains_CAS) > 2 and len(red_nodes) > 2 and len(red_edges) > 2:
            #     print('CAS动态基地剩余价值率和目标拦截率预测模块开始')
            #     ret = sim.set_status(kill_chains=kill_chains_CAS, red_nodes=red_nodes, red_edges=list(red_edges.values()))
            #
            #     print('CAS动态基地剩余价值率和目标拦截率预测模块结果：', sim.predict(x=ret))


            self.loggerInfo.emit("开始打击以下目标：" + loggerTargetStr)

            if len(Result_plan[-1]) < 8 and len(sub_plan) < 8:
                if Result_plan:
                    Result_plan[-1] = sub_plan
                else:
                    Result_plan.append(sub_plan)
            else:
                number += 1
                Result_plan[-1] = sub_plan
                Result_plan.append([])
                sub_plan = []
            if Result_plan:
                if Result_plan[0]:
                    if Result_plan[-1]:
                        # 打击方案；当前目标数；基地毁伤百分比；武器平台组合；总的目标数
                        self.Hit_result.emit(
                            Result_plan, now_targets, Des, Weapon_now, sum_TARGET, result_data_time / max(result_data_time_count, 1))
                    else:
                        self.Hit_result.emit(
                            Result_plan[:-1], now_targets, Des, Weapon_now, sum_TARGET, result_data_time / max(result_data_time_count, 1))
            result_data_time = 0
            result_data_time_count = 0
            # 监测武器和目标状态来判断是否打击成功
            for every_plan in Result_plan:
                for state in every_plan:
                    w = state[3]
                    t = state[4]
                    s = state[5]
                    if t in target_hit and s == 'x':
                        state[5] = 'T'
                    else:
                        if s == 'x':
                            if not scenario.unit_is_alive(w):
                                if w in weapon_radar_link:
                                    radar_channel[weapon_radar_link[w]] += 1
                                    del weapon_radar_link[w]
                                if scenario.unit_is_alive(t):
                                    state[5] = 'F'
                                else:
                                    state[5] = 'T'
                                    target_hit.append(t)
                                    count_hit += 1
                            else:
                                if not scenario.unit_is_alive(t):
                                    state[5] = 'F'

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
            self.loggerInfo.emit("打击信息输出结束")

            # if now_targets > 0 and sum_TARGET > 0 and len(
            #         sub_plan) > 0 and result_data_time > 0 and result_data_time_count > 0:
            #     writer.writerow(["当前目标数", str(now_targets)])
            #     writer.writerow(["基地剩余价值百分比", str(1 - Des)])
            #     writer.writerow(["目标拦截率", str(count_hit / sum_TARGET)])
            #     writer.writerow(["总目标数", str(sum_TARGET)])
            #     writer.writerow(
            #         ["算法平均运行时间", str(round(result_data_time / result_data_time_count, 4)), "单位/s"])

            time = scenario.m_Duration.split('@')
            duration = int(time[0]) * 86400 + \
                       int(time[1]) * 3600 + int(time[2]) * 60

            if scenario.m_StartTime + duration <= scenario.m_Time:

                time_list.append(result_data_time / result_data_time_count)
                baseValue_list.append(1 - Des)
                targetHit_list.append(count_hit / sum_TARGET)
                # df_eval.loc[WTAAlgorithmName + str(countRun), :] = list([1 - Des, count_hit / sum_TARGET])
                # df_eval.to_csv("record.csv", encoding='utf_8_sig')
                self.moziRunningSignal.emit()
                break
                # sys.exit(0)
            else:
                pass


# from systemPrototype.interface.src.resultData import plot_spider

# file_path = r"D:\lab\WTAProgram\systemv1.0\项目文档\成果数据\算法迭代记录.csv"
# file = open(file_path, mode='a+', newline='')
# writer = csv.writer(file)
#
# TAAlgorithmList = ["小波神经网络"]
# WTAAlgorithmList = ["基于规则的构造启发式算法",
#                     "基于比率的突出域启发式算法",
#                     "基于斜率的突出域启发式算法",
#                     "基于测验问题最优解的启发式算法",
#                     "最大边际收益启发式算法",
#                     "基于效果的武器-目标配对优化的高级输入生成算法",
#                     "近似动态规划的构造算法",
#                     "前瞻式边际贪婪构造算法", ]
# for TAAlgorithm in TAAlgorithmList:
#     for WTAAlgorithm in WTAAlgorithmList:
#         for i in range(10):
#             print("第{}次运行".format(i + 1))
#             mozi().start(TAAlgorithm, WTAAlgorithm)
#
#         p = [1 for _ in range(10)]
#         # 求baseValue_list的标准差
#         sd = np.std(baseValue_list)
#         efficiency_list, adaptability_list, performance_list = plot_spider(time_list, p, baseValue_list, sd,
#                                                                            targetHit_list)
#         writer.writerow([TAAlgorithm, WTAAlgorithm])
#         writer.writerow(["", "基地剩余价值", "目标拦截率", "效率指数", "适应性指数", "效能指数"])
#         for i in range(10):
#             writer.writerow(["第" + str(i + 1) + "次推演", baseValue_list[i], targetHit_list[i], efficiency_list[i],
#                              adaptability_list[i], performance_list[i]])
#         writer.writerow(["平均值", np.mean(baseValue_list), np.mean(targetHit_list), np.mean(efficiency_list),
#                          np.mean(adaptability_list), np.mean(performance_list)])
#         writer.writerow(["标准差", np.std(baseValue_list), np.std(targetHit_list), np.std(efficiency_list),
#                          np.std(adaptability_list), np.std(performance_list)])
#
#         time_list = []
#         baseValue_list = []
#         targetHit_list = []
