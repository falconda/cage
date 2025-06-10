# 导入numpy库
import numpy as np

rng = np.random.default_rng(123)
# 输入数据
x =  np.array([[2,3,1],
                [1,1,0],
                [0,4,2]])
u_arr = np.array([[1],[2],[3]])
print(u_arr * x)