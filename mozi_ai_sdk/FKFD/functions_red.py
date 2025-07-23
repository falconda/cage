import numpy as np
from math import sin, cos, sqrt, atan2, radians
import re
# 1

def transpose(data):
    # Case 1: 一维普通列表 → 转列向量
    if isinstance(data, list) and all(not isinstance(i, (list, np.ndarray)) for i in data):
        return [[i] for i in data]

    # Case 2: 只有一行的二维列表 → 转列向量
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        return [[i] for i in data[0]]

    # Case 3: 正常二维列表 → 转置
    if isinstance(data, list) and all(isinstance(i, list) for i in data):
        return [list(col) for col in zip(*data)]

    # Case 4: numpy array 构成的列表 → 提取每个 array 的第一行转置
    if isinstance(data, list) and all(isinstance(i, np.ndarray) for i in data):
        try:
            rows = [arr[0] for arr in data]
        except IndexError:
            raise ValueError("numpy 数组的维度必须至少为 2D，例如 shape 为 (1, N)")
        return [list(col) for col in zip(*rows)]

    raise TypeError("不支持的输入类型，请传入一维列表、二维列表、或 numpy array 列表")

# 得到距离目标最近的武器平台
def weapon_from_target(weapon_list, target_list):
    init_dis = np.inf
    index = 0
    num_weapon = [0 for _ in range(len(weapon_list))]
    for target in target_list:
        if '高超声速' in target.strName:
            continue
        else:
            for k, weapon in enumerate(weapon_list):
                dis = float(weapon.get_range_to_contact(target.strGuid))
                if dis < init_dis:
                    init_dis = dis
                    index = k
        num_weapon[index] = num_weapon[index] + 1
    return num_weapon
# 根据武器的射程输出可行性矩阵武器和目标的特性
def feasibility(weapon_range: list, dis: list, v_range: list, v, h_range: list, h, target_name: list,
                weapon_dict: dict):
    anti_name1 = '超级眼镜蛇直升机'
    anti_name2 = 'F-16DJ'
    anti_weapon = ['L1', 'L2', 'L3']
    Fij1 = np.zeros((len(weapon_range), len(h)))
    Fij2 = np.zeros((len(weapon_range), len(h)))
    Fij3 = np.ones((len(weapon_range), len(h)))
    for k, V in enumerate(weapon_range):
        every_dis = dis[k]
        for i, j in enumerate(every_dis):
            if V[1] >= j >= V[0]:
                Fij1[k][i] = 1
    for i in range(len(v_range)):
        min_v = v_range[i][0]
        max_v = v_range[i][1]
        min_h = h_range[i][0]
        max_h = h_range[i][1]
        for j in range(len(v)):
            if min_v <= v[j] <= max_v and min_h <= h[j] <= max_h:
                Fij2[i][j] = 1
    # 目标的位置
    temp_target_list = []
    for k, v in enumerate(target_name):
        if anti_name1 in v or anti_name2 in v:
            temp_target_list.append(k)
    temp_weapon_list = []
    num_weapon = []
    index = 0
    for k, v in weapon_dict.items():
        num_weapon.append(v)
        if k in anti_weapon:
            temp_weapon_list.append(index)
        index = index + 1
    if temp_weapon_list:
        for i in temp_weapon_list:
            before = sum(num_weapon[:i])
            for m in range(before, before + num_weapon[i]):
                for j in temp_target_list:
                    Fij3[m, j] = 0
    Fij = Fij1 * Fij2 * Fij3
    return Fij


# 生成打击概率矩阵
def probability_of_hit(rocket: list, max_v: list, max_l: list, target_v: list, target_l: list, base_pof: list, a,
                       limit=0.4):
    Pij1 = np.zeros((len(rocket), len(target_v)))
    Fij1 = np.ones((len(rocket), len(target_v)))
    for i, pf in enumerate(rocket):
        pof = base_pof[i]
        wv = max_v[i]
        wl = max_l[i]
        every_l = target_l[i]
        for j, tv in enumerate(target_v):
            tl = every_l[j]
            ta = a[j]
            A = ta * 0.4 * 0.8 * 0.1
            R = tl / wl
            temp_P = pof - A
            if R > 0.5:
                temp_P = pof * pf + pof * (1 - pf) * (1 - (R - pf) / (1 - pf)) - A
            if tv > wv:
                Pij1[i][j] = max(0, temp_P - 0.5)
            elif tv > 0.8 * wv:
                Pij1[i][j] = max(0, temp_P - 0.25)
            elif tv > 0.7 * wv:
                Pij1[i][j] = max(0, temp_P - 0.15)
            elif tv > 0.6 * wv:
                Pij1[i][j] = max(0, temp_P - 0.1)
            elif tv > 0.4 * wv:
                Pij1[i][j] = max(0, temp_P - 0.05)
            else:
                Pij1[i][j] = max(0, temp_P)
            if Pij1[i][j] < limit:
                Fij1[i][j] = 0
    return Pij1, Fij1
# 生成集群与系统的可行性矩阵
def Cluster_feasibility(weapon_range: list, dis: list, v_range: list, v, h_range: list, h):
    Fij1 = np.zeros((len(weapon_range), len(h)))
    Fij2 = np.zeros((len(weapon_range), len(h)))
    for k, V in enumerate(weapon_range):
        every_dis = dis[k]
        for i, j in enumerate(every_dis):
            if V[1] >= j >= V[0]:
                Fij1[k][i] = 1
    for i in range(len(v_range)):
        min_v = v_range[i][0]
        max_v = v_range[i][1]
        min_h = h_range[i][0]
        max_h = h_range[i][1]
        for j in range(len(v)):
            if min_v <= v[j] <= max_v and min_h <= h[j] <= max_h:
                Fij2[i][j] = 1
    Fij = Fij1 * Fij2
    return Fij

# 得到目标机动系数
def get_target_A(target_name, target_name_list, target_a_list):
    target_a = []
    target_class = []
    for NAME in target_name:
        list_key = []
        for name in target_name_list:
            list_key = re.findall(name, NAME)
            if list_key:
                break
        if list_key:
            key = list_key[0]
            target_a.append(target_a_list[key])
            target_class.append(key)
        else:
            target_a.append(0)
    return target_a, target_class
# 打击概率排序武器
def sort_weapon(pij, limit = 0):
    sort_list = []
    column = pij.shape[1]
    for i in range(column):
        unused = []
        weapon_list = list(pij[:, i])
        for k, v in enumerate(weapon_list):
            if v < limit:
                unused.append(k)
        sorted_id = sorted(range(len(weapon_list)), key=lambda k: weapon_list[k], reverse=True)
        for k in unused:
            sorted_id.remove(k)
        sort_list.append(sorted_id)
    return sort_list

# 得到武器组合的各种信息
def get_weapon_set(S1_weapon, weapon_pla):
    S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, S1_weapon_v = [], [], [], [], [], []
    S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_every, S1_init_num, S1_max_r = [], [], [], [], [], [], []
    for i in range(2):
        if i == 0:
            for j in S1_weapon:
                temp_dict = weapon_pla[j]
                S1_number.append(temp_dict['num_weapon'])
                S1_weapon_id.append(temp_dict['guid'])
                S1_weapon_name.append(temp_dict['name'])
                S1_weapon_every.append(temp_dict['num_storage'])
                S1_init_num.append(temp_dict['init_num'])
                S1_max_r.append(temp_dict['hit_range'][1])
        else:
            for k, j in enumerate(S1_weapon):
                temp_dict = weapon_pla[j]
                for n in range(S1_number[k]):
                    S1_weapon_range.append(temp_dict['hit_range'])
                    S1_max_l.append(temp_dict['hit_range'][1])
                    S1_weapon_h.append(temp_dict['height_range'])
                    S1_weapon_v.append(temp_dict['vel_range'])
                    S1_max_v.append(temp_dict['vel_range'][1])
                    S1_rocket.append(temp_dict['rocket'])
                    S1_pof.append(temp_dict['pof'])
    return S1_number, S1_weapon_id, S1_weapon_name, S1_weapon_range, S1_weapon_h, \
           S1_weapon_v, S1_rocket, S1_max_v, S1_max_l, S1_pof, S1_weapon_every, S1_init_num, S1_max_r


# 得到当前武器平台武器的数量
def get_current_num(info):
    if not info:
        return 0
    else:
        Num = info[0][0]
        return int(Num[:Num.rfind('x')])


# 得到当前目标的种类和数量
def get_class_num(lists):
    count_dist = dict()
    for i in lists:
        if i in count_dist:
            count_dist[i] += 1
        else:
            count_dist[i] = 1
    return count_dist
# 根据经纬度、高度计算距离 x[经度，维度，高度]
def get_distance(x, y):
    R = 6371
    lon1, lat1, alt1 = x[0], x[1], x[2]
    lon2, lat2, alt2 = y[0], y[1], y[2]
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dalt = alt2 - alt1
    a = sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c
    return sqrt(d**2 + dalt**2)
weapon_info = {
    'S1': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S2': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-2)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S3': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-3)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S4': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-4)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'M1': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40), 'height_range': (9, 24994),
           'vel_range': (0, 2963.2),
           'name': 'HQ-16B(M1-1)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'M2': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40), 'height_range': (9, 24994),
           'vel_range': (0, 2963.2),
           'name': 'HQ-16B(M1-2)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'M3': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40), 'height_range': (9, 24994),
           'vel_range': (0, 2963.2),
           'name': 'HQ-16B(M1-3)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'L1': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000001225', 'hit_range': (2, 80), 'height_range': (9, 24384),
           'vel_range': (0, 4907.8),
           'name': 'HQ-9A(L1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 12, 'original': 6},
    'S11': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S22': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-2)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S33': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-3)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S44': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-4)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'M11': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40), 'height_range': (9, 24994),
            'vel_range': (0, 2963.2),
            'name': 'HQ-16B(M2-1)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'M22': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40), 'height_range': (9, 24994),
            'vel_range': (0, 2963.2),
            'name': 'HQ-16B(M2-2)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'M33': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40), 'height_range': (9, 24994),
            'vel_range': (0, 2963.2),
            'name': 'HQ-16B(M2-3)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'L2': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000001225', 'hit_range': (2, 80), 'height_range': (9, 24384),
           'vel_range': (0, 4907.8),
           'name': 'HQ-9A(L2)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 12, 'original': 6},
    'S111': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S3-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S222': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S3-2)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S333': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S3-3)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S444': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S3-4)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'M111': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40),
             'height_range': (9, 24994), 'vel_range': (0, 2963.2),
             'name': 'HQ-16B(M3-1)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'M222': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40),
             'height_range': (9, 24994), 'vel_range': (0, 2963.2),
             'name': 'HQ-16B(M3-2)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'L3': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000001225', 'hit_range': (2, 80), 'height_range': (9, 24384),
           'vel_range': (0, 4907.8),
           'name': 'HQ-9A(L3)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 12, 'original': 6},
    'SL': {'num_weapon': 16, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'S300_1': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40),
             'height_range': (9, 24994), 'vel_range': (0, 2963.2),
             'name': 'HQ-16B(S300-1)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'S300_2': {'num_weapon': 12, 'guid': 'hsfw-dataweapon-00000000003392', 'hit_range': (2, 40),
             'height_range': (9, 24994), 'vel_range': (0, 2963.2),
             'name': 'HQ-16B(S300-2)', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 6, 'init_num': 12, 'original': 4},
    'TH1': {'num_weapon': 6, 'guid': 'hsfw-dataweapon-00000000003590', 'hit_range': (18.52, 370),
            'height_range': (30480, 609600), 'vel_range': (0, 18520),
            'name': 'TH1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 1, 'init_num': 6, 'original': 6},
    'TH2': {'num_weapon': 6, 'guid': 'hsfw-dataweapon-00000000003590', 'hit_range': (18.52, 370),
            'height_range': (30480, 609600), 'vel_range': (0, 18520),
            'name': 'TH1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 1, 'init_num': 6, 'original': 6
            },
    'GMD1': {'num_weapon': 24, 'guid': 'hsfw-dataweapon-00000000003256', 'hit_range': (7, 2453.9),
             'height_range': (100584, 1402080), 'vel_range': (0, 18520),
             'name': 'GMD1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 8, 'init_num': 24, 'original': 8
             },
    'GMD2': {'num_weapon': 24, 'guid': 'hsfw-dataweapon-00000000003256', 'hit_range': (7, 2453.9),
             'height_range': (100584, 1402080), 'vel_range': (0, 18520),
             'name': 'GMD1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 8, 'init_num': 24, 'original': 8
             }
}
weapon_info_cluster = {
    'S1': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S2': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-2)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S3': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-3)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S4': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
           'vel_range': (0, 2963.2),
           'name': 'HQ-17(S1-4)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'M1': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'M2': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'M3': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'L1': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'S11': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S22': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-2)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S33': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-3)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S44': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
            'vel_range': (0, 2963.2),
            'name': 'HQ-17(S2-4)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'M11': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'M22': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'M33': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'L2': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'S111': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S1-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S222': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S1-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S333': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S1-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'S444': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000003211', 'hit_range': (1, 9), 'height_range': (9, 6096),
             'vel_range': (0, 2963.2),
             'name': 'HQ-17(S1-1)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 8, 'init_num': 8, 'original': 2},
    'M111': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'M222': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'L3': {'num_weapon': 8, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'SL': {'num_weapon': 16, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'S300_1': {'num_weapon': 16, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'S300_2': {'num_weapon': 16, 'guid': 'hsfw-dataweapon-00000000002104', 'hit_range': (2, 209), 'height_range': (9, 38100),
           'vel_range': (0, 12239.8),
           'name': 'S-400(SL)', 'rocket': 0.5, 'pof': 0.8, 'num_storage': 4, 'init_num': 16, 'original': 8},
    'TH1': {'num_weapon': 6, 'guid': 'hsfw-dataweapon-00000000003590', 'hit_range': (18.52, 370),
            'height_range': (30480, 609600), 'vel_range': (0, 18520),
            'name': 'TH1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 1, 'init_num': 6, 'original': 6},
    'TH2': {'num_weapon': 6, 'guid': 'hsfw-dataweapon-00000000003590', 'hit_range': (18.52, 370),
            'height_range': (30480, 609600), 'vel_range': (0, 18520),
            'name': 'TH1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 1, 'init_num': 6, 'original': 6
            },
    'GMD1': {'num_weapon': 24, 'guid': 'hsfw-dataweapon-00000000003256', 'hit_range': (7, 2453.9),
             'height_range': (100584, 1402080), 'vel_range': (0, 18520),
             'name': 'GMD1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 8, 'init_num': 24, 'original': 8
             },
    'GMD2': {'num_weapon': 24, 'guid': 'hsfw-dataweapon-00000000003256', 'hit_range': (7, 2453.9),
             'height_range': (100584, 1402080), 'vel_range': (0, 18520),
             'name': 'GMD1', 'rocket': 0.5, 'pof': 0.85, 'num_storage': 8, 'init_num': 24, 'original': 8
             }
}
