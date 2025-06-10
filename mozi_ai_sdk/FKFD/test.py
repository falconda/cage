import itertools
import numpy as np


def generate_permutations(num):
    # 创建一个 0 到 num-1 的数组
    arr = list(range(num))

    # 使用 itertools.permutations 生成所有的排列
    all_permutations = list((itertools.permutations(arr)))

    return all_permutations


num = 5 # 示例，生成 0 到 2 的所有排列
permutations = generate_permutations(num)
permutations1 = np.array(permutations)

for perm in permutations1:
    print(np.array2string(perm, separator=', '))

