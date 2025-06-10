import os
import sys
import argparse
import csv
import random
import numpy as np
import timeit
import traceback

import logging
import re
from datetime import datetime

from mozi_ai_sdk.FKFD.env.env import Environment
from mozi_ai_sdk.FKFD.env import etc

from mozi_ai_sdk.FKFD.functions_blue import monitor_attack_results, monitor_aircraft_damage, pij_generate, evaluate_targets
from mozi_ai_sdk.FKFD.GA_blue import WTA_GA
parser = argparse.ArgumentParser()
parser.add_argument("--avail_ip_port", type=str, default='127.0.0.1:6060')
parser.add_argument("--platform_mode", type=str, default='eval')
parser.add_argument("--side_name", type=str, default='蓝方')
parser.add_argument("--agent_key_event_file", type=str, default=None)

#  设置墨子安装目录下bin目录为MOZIPATH，程序会自动启动墨子
os.environ['MOZIPATH'] = 'C:\\Program Files (x86)\\Mozi\\Mozi\\MoziServer\\bin'
print(os.environ['MOZIPATH'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(env):
    # 启动墨子服务器，连接墨子服务器，获取初始态势数据
    env.start()
    # 加载想定，初始化推演方
    env.reset()
    # 获取更新态势
    scenario = env.step()
    # 获取推演方，获取本方的所有数据
    # (返回的是本方的所有数据：对应API中的CSide)
    blue_side = scenario.get_side_by_name('蓝方')
    # (返回目标的字典{guid:obe.....})
    contacts_dic_blue = blue_side.get_contacts()
    # 对方目标设为敌对
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

    # 用于分配的武器名称
    keywords = ['空地战术导弹', '直接攻击炸弹', '小直径炸弹']
    flag = False
    step_count = 0

    all_attack_records = []  # 存放每一轮的 attack_records 列表
    all_attack_logs = []  # 存放每一轮的 attack_log 列表
    all_damage_logs = []    # # 存放每一轮的 damage_log 列表

    while True:
        env.step()
        step_count +=1   # 更新一步:每一步经过时长有推演倍速决定
        logging.info(f'step_count:{step_count}')
        blue_side.static_update()
        red_side.static_update()

        facilities = red_side.get_facilities()
        facilities_info = [
            [facility, facility.strGuid, facility.strName, facility.dLatitude, facility.dLongitude, facility.strDamageState]
            for facility in facilities.values()
        ]
        # logging.info(f'facilities_info:{((facilities_info))}')

        facilities_in_info = []
        for info in facilities_info:
            latitude = float(info[3])
            longitude = float(info[4])
            if 36.75 <= latitude <= 38.7 and 117.1 <= longitude <= 119.6:
                facilities_in_info.append(info)  # Guid 在索引位置 1
        # logging.info(f'facilities_in_info:{((facilities_in_info))}')

        contacts_dic_blue = blue_side.get_contacts()
        # 将目标信息转为列表结构，每个元素包含：[目标类对象, GUID, 名称, 纬度, 经度]
        targets_info = [
            [target, target.strGuid, target.strName, target.dLatitude, target.dLongitude]
            for target in contacts_dic_blue.values()
        ]
        # logging.info(f'targets_info:{targets_info}')

        # 实时筛选：区域内目标 Guid 列表
        targets_in_info = []
        for info in targets_info:
            latitude = float(info[3])
            longitude = float(info[4])
            if 36.75 <= latitude <= 38.7 and 117.1 <= longitude <= 119.6:
                targets_in_info.append(info)  # Guid 在索引位置 1
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
                logging.info(f'飞机{item[2]}没有武器{item[3]}返回基地')

        # 攻击逻辑：生成配对并打击，只执行一次
        # 提前分配，然后是导弹数都变化后再分配
        # 15倍速，完成第一轮打击
        trigger = 50

        # 如何对结果进行结算？定时实现
        '''
        先结算，然后再产生新的方案，以及对应的attack_log
        '''
        # if step_count == 1 or (step_count > 300 and step_count % 101 == 0):
        if step_count > trigger and (step_count-trigger) % 30 == 0:
            # 毁伤情况记录
            attack_log = monitor_attack_results(attack_records, facilities_in_info)
            # 损伤情况记录
            damage_log = monitor_aircraft_damage(attack_records, acs_assign_info)
            # 保存到总集合中
            all_attack_logs.append(attack_log)
            all_damage_logs.append(damage_log)
            logging.info(f'attack_logs:{all_attack_logs}')
            logging.info(f'damage_logs:{all_damage_logs}')

        if step_count == 1 or step_count > trigger and (step_count-trigger) % 30 == 0:
            #  配对产生：设计算法和模型
            # if len(weapon_num) > 0:
            pij = pij_generate(targets_in_info, acs_assign_weapon)
            value = evaluate_targets(targets_in_info, facilities_in_info)
            logging.info(f'武器长度{len(acs_assign_weapon)}, 目标长度{len(targets_in_info)}')
            # 初始化算法
            solver = WTA_GA(pij, value, weapon_num, pop_size=30, generations=100)
            plan, b_fitness = solver.evolve()
            logging.info(f'产生plan{plan}, 对应适应度{b_fitness}')

            # 依据打击方案，记录分配情况
            attack_records = []
            for i in range(len(plan)):
                row = plan[i]

                ac, guid, name, weapon_name, count, wid = acs_assign_weapon[i]

                for j, num in enumerate(row):
                    if num > 0 and j < len(targets_in_info):
                        target_obj, target_guid, target_name, target_lat, target_lon = targets_in_info[j]

                        ac.manual_attack(target_guid, wid, num)
                        logging.info(f"飞机 {name}（编号: {i}） 使用武器 {weapon_name}（数量: {num}） 攻击目标 {target_name}（编号: {j}）")
                        attack_records.append([
                            ac, guid, name,
                            target_obj, target_guid, target_name, target_lat, target_lon,
                            weapon_name, wid, num
                        ])
                        break  # 只处理一个非零元素

            all_attack_records.append(attack_records)

        # 根据发射的武器确定取消积压命令和避免浪费
        # 没分配成功的要手动取消
        # unit_drop_target_contact(t)



        if step_count == 450:
            logging.info(f'all_attack_records:{all_attack_records}')
            logging.info(f'all_attack_logs:{all_attack_logs}')
            logging.info(f'all_damage_logs:{all_damage_logs}')


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
        print('开发模式')
        env = Environment(ip=etc.SERVER_IP, port=etc.SERVER_PORT, platform=etc.PLATFORM,
                          scenario_name=etc.SCENARIO_NAME, simulate_compression=etc.SIMULATE_COMPRESSION,
                          duration_interval=etc.DURATION_INTERVAL, synchronous=etc.SYNCHRONOUS, app_mode=etc.app_mode)

        run(env)

try:
    main()
except Exception as e:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'error_{timestamp}.log'
    with open(log_filename, 'w', encoding='utf-8') as error_file:
        traceback.print_exc(file=error_file)
    sys.exit()
