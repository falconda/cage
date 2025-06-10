import csv
import random
import numpy as np
import time
from PySide2.QtCore import QObject, Signal, Slot, QCoreApplication
import os
import sys
import pandas as pd
import ast
from Heuristic_WTA import Heuristic_WTA

sys.path.append('D:\\Workplace\\moziai\\mozi_ai_sdk\\FKFD\\WTA')
from IGWO_WTA import IGWO_WTA
from DF_WTA import DF_WTA
from TS_WTA import TS_WTA

csv_file = f'D:\Workplace\moziai\mozi_ai_sdk\FKFD\WTA\selectWTA.csv'
output = f'output1024_DF.csv'

def read_row(file_path, n, i):
    start_row = n * 10 + i
    end_row = (n + 1) * 10 + i

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = [row for idx, row in enumerate(reader) if start_row <= idx < end_row]
        #
        # cleaned_data1 = [[item.replace('         ', ' ') for item in sublist] for sublist in data]
        # cleaned_data2 = [[item.replace('       ', ' ') for item in sublist] for sublist in cleaned_data1]
        # cleaned_data3 = [[item.replace('      ', ' ') for item in sublist] for sublist in cleaned_data2]
        # cleaned_data4 = [[item.replace('     ', ' ') for item in sublist] for sublist in cleaned_data3]
        # cleaned_data5 = [[item.replace('    ', ' ') for item in sublist] for sublist in cleaned_data4]
        # cleaned_data6 = [[item.replace('   ', ' ') for item in sublist] for sublist in cleaned_data5]
        # cleaned_data7 = [[item.replace('  ', ' ') for item in sublist] for sublist in cleaned_data6]
        # cleaned_data8 = [[item.replace(' ', ',') for item in sublist] for sublist in cleaned_data7]
        # # 处理多余逗号
        # cleaned_data9 = [[item.replace(',,', ',') for item in sublist] for sublist in cleaned_data8]
        #
        # # 转换为浮点数
        # final_data = []
        # for sublist in cleaned_data9:
        #     final_sublist = []
        #     for item in sublist:
        #         if item.startswith('[[') and item.endswith(']]'):
        #             # 将字符串转为实际的列表
        #             item = item.replace(',,', ',')  # 再次确保没有多余的逗号
        #             inner_list = eval(item)  # 注意：使用 eval 有风险，确保数据来源安全
        #             final_sublist.append([[float(i) for i in inner_sublist] for inner_sublist in inner_list])
        #         else:
        #             final_sublist.append(float(item))  # 转换其他项
        #     final_data.append(final_sublist)
        #
        # # print(final_data)
        # final_data1 = final_data[0]
        # # print(final_data1)
        data1 = []
        for i in range(7):
            row = data[0][i]
            row = row.replace("\n ", " ")
            row = row.replace("     ", " ")
            row = row.replace("    ", " ")
            row = row.replace("   ", " ")
            row = row.replace("  ", " ")
            row = row.replace(" ", ",")
            row = row.replace(",,", ",")
            row_list = ast.literal_eval(row)
            data1.append(row_list)
        final_data1 = data1

    return final_data1  # 返回读取到的行数据

n=0
np.set_printoptions(threshold=sys.maxsize)
for i in range(101):
    row_data = read_row(csv_file, n, i)
    weapon_num = row_data[0]
    target_num = row_data[1]
    Vec_Wei = np.array(row_data[2])
    Mar_pij = np.array(row_data[3])
    Fij = np.array(row_data[4])
    qjk = np.array(row_data[5])
    V_a = np.array(row_data[6])
    start_time = time.time()
    al = DF_WTA(weapon_num, target_num, Vec_Wei, Mar_pij, Fij, qjk, V_a)
    best_plan = al.run()
    end_time = time.time()
    ttime = end_time-start_time
    compute_fitness = Heuristic_WTA(weapon_num, target_num, Vec_Wei, Mar_pij, Fij, qjk, V_a)
    fitness = round(compute_fitness.compute_fitness(best_plan, a=0.8, b=0.2), 4)
    print(i)
    print(best_plan)
    print(fitness)
    print(ttime)
    with open(output, 'a', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([weapon_num, target_num, Vec_Wei, Mar_pij, Fij, qjk, V_a, fitness, ttime])