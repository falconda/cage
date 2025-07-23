import numpy as np
from typing import List, Dict
import re
import math
import torch
#1234567890
def monitor_attack_results(attack_records:list, facilities_info:list, coord_tol=0.0001):
    """
    监控打击结果，更新损伤信息（使用经纬度配对目标）。适配列表结构 attack_records。

    参数：
    - attack_records: list[list]，每条记录结构为：
        [ac_obj, ac_guid, ac_name, target_obj, target_guid, target_name, target_lat, target_lon, weapon_name, weapon_dbid, ...]
    - result_log: list[list]，保存日志的结果，每项为：
        [ac_name, ac_guid, target_name, target_guid, damage_before, damage_after, damage_delta, hit]
    - facilities_info: list[list]，每个设施：[facility_obj, guid, name, lat, lon, damage]
    - coord_tol: float，经纬度匹配误差容忍值
    """
    result_log = []
    for record in attack_records:
        ac_guid = record[1]
        ac_name = record[2]
        target_guid = record[4]
        target_name = record[5]
        target_lat = record[6]
        target_lon = record[7]
        weapon_name = record[8]
        weapon_dbid = record[9]
        weapon_num = record[10]

        current_damage = None

        # 用经纬度在设施中查找匹配项
        for fac_entry in facilities_info:
            fac_lat, fac_lon = fac_entry[3], fac_entry[4]
            if abs(fac_lat - target_lat) <= coord_tol and abs(fac_lon - target_lon) <= coord_tol:
                try:
                    current_damage = float(fac_entry[5])
                except:
                    current_damage = 0.0
                break

        # 找不到对应设施，说明已摧毁
        if current_damage is None:
            current_damage = 1.0

        # 初次添加监控字段
        if len(record) == 11:
            record.append(0)  # [11] damage_before
            record.append(current_damage)  # [12] damage_after
            if current_damage > 0:
                record.append(True)           # [13] hit_success
            else:
                record.append(False)
            damage_before = 0
            damage_after = current_damage
            damage_delta = damage_after - damage_before
            hit = damage_delta > 0.001

        else:
            damage_before = record[11]
            damage_after = current_damage
            damage_delta = damage_after - damage_before
            hit = damage_delta > 0.001  # 命中判定阈值

            # 更新记录
            record[11] = damage_after
            record[12] = hit

        # 日志输出：不管命中与否都记录
        result_log.append([
            ac_name,
            weapon_name,
            weapon_num,
            target_name,
            round(damage_before, 3),
            round(damage_after, 3),
            round(damage_delta, 3),
            hit
        ])

    return result_log


def monitor_aircraft_damage(attack_records:list,  acs_assign_info:list):
    """
    监控每架被攻击飞机的损伤变化，更新记录。
    - attack_records: 当前轮攻击分配记录，每行为 [ac, guid, name, target_obj, target_guid, ...]
    - result_log: 当前轮的监控日志（list of rows）
    - acs_assign_info: 当前存活飞机信息，每行为 [ac_obj, guid, name, lat, lon, strDamageState]
                      后续将扩展 [damage_before, damage_after, hit]
    """

    result_log = []
    # 构造 guid -> ac_record 映射
    # attack_records里面的飞机类是 acs_assign_info的子集，肯定会在里面，但是被打下来肯定就没有
    acs_dict = {ac[1]: ac for ac in acs_assign_info}

    for rec in attack_records:
        ac_guid = rec[1]
        ac_name = rec[2]

        # 默认假设被击落
        current_damage = 1.0

        if ac_guid in acs_dict:
            ac_record = acs_dict[ac_guid]

            # 提取当前损伤状态
            try:
                current_damage = float(ac_record[5])
            except:
                current_damage = 0.0  # 如果格式异常，认为未受损

            # 添加监控字段：若第一次监控则追加字段
            if len(ac_record) < 9:
                damage_before = 0.0
                ac_record.append(damage_before)        # [6]
                ac_record.append(current_damage)       # [7]
                ac_record.append(current_damage > 0.001)  # [8]
            else:
                damage_before = ac_record[7]
                ac_record[6] = damage_before
                ac_record[7] = current_damage
                ac_record[8] = current_damage - damage_before > 0.001
        else:
            # 飞机被打下了
            damage_before = 0.0  # 若之前未出现，认为从0开始受损

        damage_delta = current_damage - damage_before
        hit = damage_delta > 0.001

        # 加入本轮监控日志
        result_log.append([
            ac_name,
            round(damage_before, 3),
            round(current_damage, 3),
            round(damage_delta, 3),
            hit
        ])

    return result_log

def get_red_damage(facilities_in):
    names = []
    damage_state = []

    for target in facilities_in:
        name = target.strName
        damage = float(target.strDamageState) / 100
        # 加入输出列表
        names.append(name)
        damage_state.append(damage)

    return [names, damage_state]

# 全局定义武器名称 -> 打击概率字典
weapon2pij_dict = {
    'AGM-65G2型“小牛”空地战术导弹': 0.9,
    'GBU-53/B 小直径炸弹II': 0.95,
    'GBU-38(V)1/B联合直接攻击炸弹': 0.95,
    '战斧-3000亚声速反舰/对陆导弹': 0.85,
    # 你可以在这里添加更多武器
}

def pij_generate(targets_in_info: list, acs_assign_weapon: list):
    '''
    产生打击概率矩阵，基于武器名称对应的固定概率poH

    :param targets_in_info: 在范围区域内的目标信息list
    :param acs_assign_weapon: 可分配武器的飞行单位，每个元素是武器名称（str）
    :return: pij打击概率矩阵 (n_ac x n_target)
    '''
    n_ac = len(acs_assign_weapon)
    n_target = len(targets_in_info)

    pij = np.zeros((n_ac, n_target))

    for i in range(n_ac):
        weapon_name = acs_assign_weapon[i][3]
        base_prob = weapon2pij_dict.get(weapon_name, 0.5)  # 默认概率为 0.5
        pij[i, :] = base_prob
        # for j in range(n_target):
        #     # noise = np.random.normal(loc=0.0, scale=0.02)  # 加一点随机扰动
        #     pij[i, j] = np.clip(base_prob + noise, 0.0, 1.0)

    return pij

import numpy as np


# ----------能力指标计算函数 ----------

target_input_dict= {
    'HQ-17': {
        'v': 0, 'v_min': 0, 'v_max': 60,
        'type': '防空火力类',
        'R': 16.668, 'R_min': 0, 'R_max': 387.068,  # 用的公里
        'S': 30, 'S_min': 0, 'S_max': 30,   # 用的60s/发射间隔
        'K': 15, 'K_min': 0, 'K_max': 180,  # 用的战斗部信息
        'ammo_type': '制导弹',
        'armor_type': '防空导弹/雷达',
        'mission': '主攻',
        'is_visible': True,
        'weather': '好',
        'importance_desc': '大', 'importance_certainty': '完全确定',
        'urgency_desc': '较大', 'urgency_certainty': '相对确定',
    },
    'HQ-16B': {
        'v': 0, 'v_min': 0, 'v_max': 60,
        'type': '防空火力类',
        'R': 74.08, 'R_min': 0, 'R_max': 387.068,
        'S': 20, 'S_min': 0, 'S_max': 30,
        'K': 70, 'K_min': 0, 'K_max': 180,
        'ammo_type': '制导弹',
        'armor_type': '防空导弹/雷达',
        'mission': '主攻',
        'is_visible': True,
        'weather': '好',
        'importance_desc': '大', 'importance_certainty': '完全确定',
        'urgency_desc': '大', 'urgency_certainty': '相对确定',
    },
    'HQ-9A': {
        'v': 0, 'v_min': 0, 'v_max': 60,
        'type': '防空火力类',
        'R': 148.16, 'R_min': 0, 'R_max': 387.068,
        'S': 8, 'S_min': 0, 'S_max': 30,
        'K': 180, 'K_min': 0, 'K_max': 180,
        'ammo_type': '制导弹',
        'armor_type': '防空导弹/雷达',
        'mission': '主攻',
        'is_visible': True,
        'weather': '好',
        'importance_desc': '大', 'importance_certainty': '完全确定',
        'urgency_desc': '很大', 'urgency_certainty': '相对确定',
    },
    'S-400E': {
        'v': 0, 'v_min': 0, 'v_max': 60,
        'type': '防空火力类',
        'R': 387.068, 'R_min': 0, 'R_max': 387.068,
        'S': 30, 'S_min': 0, 'S_max': 30,
        'K': 123, 'K_min': 0, 'K_max': 180,
        'ammo_type': '制导弹',
        'armor_type': '自行火炮类',
        'mission': '主攻',
        'is_visible': True,
        'weather': '好',
        'importance_desc': '大', 'importance_certainty': '完全确定',
        'urgency_desc': '很大', 'urgency_certainty': '相对确定',
    },
    '雷达': {
        'v': 0, 'v_min': 0, 'v_max': 60,
        'type': '火力支援类',
        'R': 0, 'R_min': 0, 'R_max': 387.068,
        'S': 0, 'S_min': 0, 'S_max': 30,
        'K': 0, 'K_min': 0, 'K_max': 180,
        'ammo_type': '无',
        'armor_type': '防空导弹、雷达',
        'mission': '助攻',
        'is_visible': True,
        'weather': '好',
        'importance_desc': '大', 'importance_certainty': '完全确定',
        'urgency_desc': '较大', 'urgency_certainty': '相对确定',
    },
    '基地': {
        'v': 0, 'v_min': 0, 'v_max': 60,
        'type': '指挥保障类',
        'R': 0, 'R_min': 0, 'R_max': 387.068,
        'S': 0, 'S_min': 0, 'S_max': 20,
        'K': 0, 'K_min': 0, 'K_max': 180,
        'ammo_type': '无',
        'armor_type': '混凝土工事类',
        'mission': '指挥',
        'is_visible': True,
        'weather': '好',
        'importance_desc': '极大', 'importance_certainty': '完全确定',
        'urgency_desc': '极大', 'urgency_certainty': '相对确定',
    },
}


def calc_mobility(v_i: float, v_min: float, v_max: float) -> float:
    """计算机动能力归一化值"""
    return (v_i - v_min) / 2*(v_max - v_min) + 0.5 if v_max != v_min else 0.5

def calc_comm_ability(target_type: str) -> float:
    """根据目标类型量化通信能力"""
    comm_table = {
        '指挥保障类': 1.0,
        '防空火力类': 0.9,
        '火力支援类': 0.7,
        '地面突击类': 0.6,
        '工程障碍类': 0.5
    }
    return comm_table.get(target_type, 0.5)

def calc_firepower(R, R_min, R_max, S, S_min, S_max, K, K_min, K_max, ammo_type, weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    AMMO_TYPE_SCORE = {
        '普通榴弹': 0.7,
        '穿甲弹': 0.75,
        '制导弹': 0.85,
        '末敏弹': 0.9,
        '集束弹': 0.95,
        '战术导弹': 1.0
    }
    """计算打击能力，四因素加权"""
    i_range = 0.5 + 0.5 * (R - R_min) / (R_max - R_min) if R_max != R_min else 0.5
    i_rate = 0.5 + 0.5 * (S - S_min) / (S_max - S_min) if S_max != S_min else 0.5
    i_caliber = 0.5 + 0.5 * (K - K_min) / (K_max - K_min) if K_max != K_min else 0.5
    ammo_score = AMMO_TYPE_SCORE.get(ammo_type, 0.5)
    return weights[0] * i_range + weights[1] * ammo_score + weights[2] * i_rate + weights[3] * i_caliber

def calc_vulnerability(target_type: str) -> float:
    """根据目标类型量化易损性"""
    vuln_table = {
        '装甲车/坦克类': 0.1,
        '混凝土工事类': 0.2,
        '牵引火炮类': 0.3,
        '自行火炮类': 0.4,
        '防空导弹/雷达': 0.5
    }
    return vuln_table.get(target_type, 0.3)

# ---------- 二、动态指标 ----------

def calc_distance_threat(R, R_recon, R_effective, R_max) -> float:
    """计算距离威胁程度"""
    if R > R_recon:
        return 0.0
    elif R <= R_effective:
        return 1.0
    elif R <= R_max:
        return (R_max - R) / (R_max - R_effective)
    else:
        return 0.0

def calc_attack_task(task: str) -> float:
    """量化攻击任务"""
    task_table = {
        '指挥': 1.0,
        '主攻': 1.0,
        '助攻': 0.9,
        '支援': 0.8,
        '穿插': 0.7,
        '掩护': 0.6,
        '防御': 0.5
    }
    return task_table.get(task, 0.5)

# ---------- 三、环境指标 ----------

def calc_visibility(is_visible: bool, weapon_type: str, m1=0.2, m2=0.8) -> float:
    """计算通视情况的威胁值"""
    if weapon_type in ['防空火力类', '间瞄武器']:
        return np.random.uniform(m2, 1.0)
    return np.random.uniform(m2, 1.0) if is_visible else np.random.uniform(0, m1)

def calc_weather(weather_level: str) -> float:
    """气象条件影响程度"""
    table = {
        '很好': 1.0,
        '好': 0.9,
        '较好': 0.7,
        '一般': 0.6,
        '较差': 0.5,
        '差': 0.3,
        '很差': 0.1
    }
    return table.get(weather_level, 0.5)

# ---------- 四、态势指标（模糊语言 + 区间转化） ----------

fuzzy_eval_table = {
    "极大": (1.0, 0.0), "很大": (0.9, 0.05), "大": (0.8, 0.1), "较大": (0.7, 0.15),
    "稍大": (0.6, 0.2), "中等": (0.5, 0.5), "稍小": (0.4, 0.4), "较小": (0.3, 0.55),
    "小": (0.2, 0.7), "很小": (0.1, 0.85), "极小": (0.0, 1.0)
}

certainty_table = {
    "完全确定": (0.8, 1.0), "相对确定": (0.6, 0.8), "一般": (0.4, 0.6),
    "不太确定": (0.2, 0.4), "不确定": (0.0, 0.2)
}


def fuzzy_to_real(t: float, f: float, c_l: float, c_u: float) -> float:
    """融合模糊语言与确定程度后转化为实数"""
    return t * (c_l + c_u) / 2 + (1 - f) * (1 - (c_l + c_u) / 2)


def calc_fuzzy_index(desc: str, certainty: str) -> float:
    """态势指标：重要性或紧迫性"""
    t, f = fuzzy_eval_table[desc]
    c_l, c_u = certainty_table[certainty]
    return fuzzy_to_real(t, f, c_l, c_u)


def compute_static_index(target_static: Dict) -> Dict[str, float]:
    index = {}

    index['I1_机动能力'] = calc_mobility(target_static['v'], target_static['v_min'], target_static['v_max'])
    index['I2_通信能力'] = calc_comm_ability(target_static['type'])
    index['I3_打击能力'] = target_static.get('I3_打击能力', calc_firepower(
        target_static['R'], target_static['R_min'], target_static['R_max'],
        target_static['S'], target_static['S_min'], target_static['S_max'],
        target_static['K'], target_static['K_min'], target_static['K_max'],
        target_static['ammo_type']
    ))  # <<< 改为传类型
    index['I4_易损性'] = calc_vulnerability(target_static['armor_type'])
    index['I6_攻击任务'] = calc_attack_task(target_static['mission'])
    index['I7_通视情况'] = calc_visibility(target_static['is_visible'], target_static['type'])
    index['I8_气象条件'] = calc_weather(target_static['weather'])
    index['I9_重要性'] = calc_fuzzy_index(target_static['importance_desc'], target_static['importance_certainty'])
    index['I10_紧迫性'] = calc_fuzzy_index(target_static['urgency_desc'], target_static['urgency_certainty'])

    return index


def compute_dynamic_index(target_name: str, mission: str, damage_dict: Dict[str, float]) -> float:
    damage = damage_dict.get(target_name, 0.0)  # 默认未损为0
    mission_score = calc_attack_task(mission)
    return 0.5 * (1 - damage) + 0.5 * mission_score


def extract_type_from_name(name: str) -> str:
    """根据目标完整名称提取其类型关键字"""
    # 优先匹配特殊类型
    if '基地' in name:
        return '基地'
    if '雷达' in name:
        return '雷达'

    # 否则尝试默认提取括号前内容（如 HQ-17（S1-1））
    match = re.match(r"([^\（\(]+)", name)
    return match.group(1).strip() if match else name.strip()


def evaluate_targets(targets_info, facilities_in_info):

    def build_damage_dict(facilities_in_info):
        """构建目标名称到毁伤值的映射字典"""
        damage_dict = {}
        for facility in facilities_in_info:
            facility_name = facility[2]  # strName
            damage_str = facility[5]  # strDamageState，形如 '0.6'
            try:
                damage_value = float(damage_str)/100

            except:
                damage_value = 0.0  # 默认值
            damage_dict[facility_name] = damage_value
        return damage_dict

    damage_dict = build_damage_dict(facilities_in_info)

    threat_array = []

    for entry in targets_info:
        target_obj = entry[0]
        full_name = entry[2]
        type_key = extract_type_from_name(full_name)

        if type_key in target_input_dict:
            static_input = target_input_dict[type_key]
            static_input = static_input.copy()  # 防止污染原始模板
            static_input['name'] = full_name  # 添加名字用于动态映射

            # 静态部分
            static_index = compute_static_index(static_input)

            # 动态部分：I5_动态威胁度
            dynamic_i5 = compute_dynamic_index(
                target_name=full_name,
                mission=static_input['mission'],
                damage_dict=damage_dict
            )
            static_index['I5_动态威胁度'] = dynamic_i5

            # 加权计算总威胁值
            weights = {
                'I1_机动能力': 0.11401,
                'I2_通信能力': 0.11624,
                'I3_打击能力': 0.082321,
                'I4_易损性': 0.07118,
                'I5_动态威胁度': 0.09044,
                'I6_攻击任务': 0.07451,
                'I7_通视情况': 0.04463,
                'I8_气象条件': 0.13653,
                'I9_重要性': 0.12137,
                'I10_紧迫性':  0.14878
            }

            threat_score = sum(static_index[k] * weights[k] for k in weights)
            threat_array.append(threat_score)
        else:
            print(f"[警告] 无法识别目标类型：{type_key}")
            threat_array.append(0.0)

    return threat_array

def extract_targets_attributes(facilities_in):
    '''
        提取打击平台的属性信息：
        - 名称
        - 经纬度
        - 型号
        - 分类（武器平台/基地/雷达）
        - 最大射程
        - 弹药类型 + 映射值
        - 武器部能力
    :param targets_in: 红方设施类的集合list，contact类
    :param facilities: 红方设施类的集合list，facility类
    :return: 数据库的红方部分的信息
    '''


    type_category_map = {
        'HQ-17': 0,
        'HQ-16B': 0,
        'HQ-9A': 0,
        'S-400E': 0,
        '基地': 1,
        '雷达': 2
        # ……其他型号可以继续添加
    }

    ammo_type_map = {
        '普通榴弹': 0,
        '穿甲弹': 1,
        '制导弹': 2,
        '末敏弹': 3,
        '集束弹': 4,
        '战术导弹': 5,
        '未知': -1
    }

    names = []
    type_categories = []
    positions = []
    damage_state=[]
    strike_ranges = []
    ammo_type_ids = []
    weapon_capabilities = []

    for target in facilities_in:
        name = target.strName
        lat = target.dLatitude
        lon = target.dLongitude

        damage = float(target.strDamageState)/100


        type_name = extract_type_from_name(name)
        type_category = type_category_map.get(type_name, -1)  # 若找不到则默认未知

        # 默认值
        R_max = 0
        ammo_type = '未知'
        ammo_type_id = -1
        K_max = 0

        if type_name in target_input_dict:
            data = target_input_dict[type_name]
            R_max = data.get('R', 0)
            ammo_type = data.get('ammo_type', '未知')
            ammo_type_id = ammo_type_map.get(ammo_type, -1)
            K_max = data.get('K', 0)

        # 加入输出列表
        names.append(name)
        positions.append([lat, lon])
        type_categories.append(type_category)
        damage_state.append(damage)
        strike_ranges.append(R_max)
        ammo_type_ids.append(ammo_type_id)
        weapon_capabilities.append(K_max)

    return [names, type_categories, positions, damage_state, strike_ranges, ammo_type_ids, weapon_capabilities]

def extract_target_encoded_attributes(facilities_in):
    """
    提取每个目标的属性，分别存入不同的列表变量中。
    返回所有这些列表，顺序与字段名保持一致。
    """

    # 字符映射字典
    type_map = {
        '指挥保障类': 1,
        '防空火力类': 2,
        '火力支援类': 3,
        '地面突击类': 4,
        '工程障碍类': 5,
    }

    ammo_type_score = {
        '普通榴弹': 1,
        '穿甲弹': 2,
        '制导弹': 3,
        '末敏弹': 4,
        '集束弹': 5,
        '战术导弹': 6,
        '无': 0
    }

    armor_type_score = {
        '装甲车/坦克类': 1,
        '混凝土工事类': 2,
        '牵引火炮类': 3,
        '自行火炮类': 4,
        '防空导弹/雷达': 5,
        '防空导弹、雷达': 6,
    }

    mission_score = {
        '指挥': 1,
        '主攻': 2,
        '助攻': 3,
        '支援': 4,
        '穿插': 5,
        '掩护': 6,
        '防御': 7,
    }

    weather_score = {
        '很好': 1,
        '好': 2,
        '较好': 3,
        '一般': 4,
        '较差': 5,
        '差': 6,
        '很差': 7
    }

    # 模糊值计算函数
    def fuzzy_val(desc, certainty):
        if desc in fuzzy_eval_table and certainty in certainty_table:
            t, f = fuzzy_eval_table[desc]
            c_l, c_u = certainty_table[certainty]
            return fuzzy_to_real(t, f, c_l, c_u)
        return 0.5

    # 每类属性一个列表
    type_names = []
    lats = []
    lons = []
    Rs = []
    Ss = []
    Ks = []
    names = []
    type_codes = []
    ammo_type_codes = []
    armor_type_codes = []
    mission_codes = []
    is_visible_codes = []
    weather_codes = []
    damages = []
    importance_scores = []
    urgency_scores = []

    # 遍历设施，提取属性
    for target in facilities_in:
        name = target.strName
        lat = target.dLatitude
        lon = target.dLongitude
        type_name = extract_type_from_name(name)
        base_attr = target_input_dict.get(type_name, {})

        # 添加各属性值
        names.append(name)
        # type_names.append(type_name)
        lats.append(lat)
        lons.append(lon)
        Rs.append(base_attr.get('R', 0.0))
        Ss.append(base_attr.get('S', 0.0))
        Ks.append(base_attr.get('K', 0.0))

        type_codes.append(type_map.get(base_attr.get('type', ''), -1))
        ammo_type_codes.append(ammo_type_score.get(base_attr.get('ammo_type', ''), -1))
        armor_type_codes.append(armor_type_score.get(base_attr.get('armor_type', ''), -1))
        mission_codes.append(mission_score.get(base_attr.get('mission', ''), -1))
        is_visible_codes.append(1 if base_attr.get('is_visible', True) else -1)
        weather_codes.append(weather_score.get(base_attr.get('weather', ''), -1))

        # 毁伤
        try:
            damages.append(float(target.strDamageState) / 100)
        except Exception:
            damages.append(0.0)

        importance_scores.append(fuzzy_val(base_attr.get('importance_desc', ''), base_attr.get('importance_certainty', '')))
        urgency_scores.append(fuzzy_val(base_attr.get('urgency_desc', ''), base_attr.get('urgency_certainty', '')))

    return [names, lats, lons, Rs, Ss, Ks,
        type_codes, ammo_type_codes, armor_type_codes,
        mission_codes, is_visible_codes, weather_codes,
        damages, importance_scores, urgency_scores]

def apply_assignment_with_limits(plan:list, tij:list, weapon_num:list):
    """
    根据遗传算法输出的配对矩阵 plan，结合 tij 和 weapon_num 得到实际分配矩阵
    :param plan: [n_units x n_targets] 0-1 配对矩阵
    :param tij: [n_units x n_targets] 所需武器数量矩阵
    :param weapon_num: [n_units] 每个单位的武器总数
    :return: [n_units x n_targets] 实际分配矩阵
    """
    n_units, n_targets = np.array(plan).shape
    tij = np.array(tij)
    weapon_num = np.array(weapon_num)
    actual = np.zeros_like(plan, dtype=int)

    for i in range(n_units):
        for j in range(n_targets):
            if plan[i][j] == 1:
                assign_qty = min(tij[i][j], weapon_num[i])  # 若多个目标时，这里还要考虑剩余量
                actual[i][j] = assign_qty
                weapon_num[i] -= assign_qty  # 减掉本次消耗
                if weapon_num[i] <= 0:
                    break  # 当前单位无剩余武器，跳出

    return actual

platform_weapon_capability = {
    'GBU-38(V)1/B联合直接攻击炸弹': 87/136,
    'AGM-65G2型“小牛”空地战术导弹': 136/136,
}

target_defense_value = {
    'HQ-17': 0.5,
    'HQ-16B': 0.5,
    'HQ-9A': 0.5,
    'S-400E': 0.5,
    '雷达': 0.5,
    '基地': 0.8

}

def build_tij_matrix(acs_assign_weapon, targets_in_info):
    n_units = len(acs_assign_weapon)
    n_targets = len(targets_in_info)
    tij = np.zeros((n_units, n_targets), dtype=int)

    for i, (ac, guid, name, weapon_name, count, wid) in enumerate(acs_assign_weapon):

        attack_power = platform_weapon_capability.get(weapon_name, 1)  # 避免除0

        for j, (target, guid_t, name_t, lat, lon) in enumerate(targets_in_info):
            name_type = extract_type_from_name(name_t)
            defense = target_defense_value.get(name_type, 1)
            tij[i][j] = math.ceil( 2*defense / attack_power )

    return tij.tolist()