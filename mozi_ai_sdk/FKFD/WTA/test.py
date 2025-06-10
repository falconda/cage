import random
import numpy as np

sort_indices = [2.0, 3.0]
integer_list = map(lambda x: int(x), sort_indices)
print(list(integer_list))

'''
matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1]])
for i in range(3):
    # 存储选中的数的位置
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    selected_numbers = np.zeros(min(num_rows, num_cols))
    selected_locations = []

    random_indices = np.zeros(num_rows)
    if num_rows > num_cols:  # 行数大于列数
        # 生成随机索引列表
        random_indices = np.random.choice(num_rows, num_cols, replace=False)
        print("random_indices", random_indices)
        random_indices = np.array(sorted(random_indices))
        print("random_indices", random_indices)
        # 根据索引获取对应的行
        new_matrix = matrix[random_indices]
        # print("new_matrix", new_matrix)

    elif num_rows == num_cols:  # 行数等于列数
        random_indices = np.array([x for x in range(num_rows)])
        # np.random.shuffle(random_indices)

    else:  # 行数小于列数
        random_indices = np.array([x for x in range(num_rows)])
        # np.random.shuffle(random_indices)

    # col_select_list = np.array([x for x in range(num_cols)])

    # 对每一行进行操作
    while True:
        col_select_list = np.array([x for x in range(num_cols)])
        for index, element in enumerate(random_indices):
            row = matrix[element]
            # 随机选择一个不为0的数，并确保它来自不同的列
            rand_col = np.random.choice(col_select_list)  # 从列数的列表中随机选一列
            # 选择的列对应元素不为0，且：不在保存的所有列中or第一次选:保存这次选的元素的行与列
            selected_numbers[index] = row[rand_col]
            selected_locations.append((element, rand_col))
            # 删除候选列表中的列值，下一次不会所有到
            col_select_list = np.delete(col_select_list, np.where(col_select_list == rand_col))

        if 0 in selected_numbers:
            print('有0元素，返回最开始循环')
            # 清空保存 因为selected_numbers里面一直会有0，会一直进这个循环
            continue
        else:
            break

        # 保存产生的列的位置rand_col， 对应元素值row[rand_col]

        # 这一行其他元素置零 好像不用这样操作也行
        for i in range(len(row)):
            if i != rand_col:
                row[i] = 0

    # 创建一个空列表来存储选中的数
    selected_numbers_list = np.full(matrix.shape[0], -1)

    # 将选中的数添加到列表中
    for row, col in selected_locations[-3:]:
        selected_numbers_list[row] = col
    print('现在是第{}次循环'.format(i))
    print('selected_numbers', selected_numbers)
    print('selected_locations', selected_locations)
    print('selected_numbers_list', selected_numbers_list)
    print("\n")

'''


