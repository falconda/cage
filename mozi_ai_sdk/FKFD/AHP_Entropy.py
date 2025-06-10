from dataProcess_AHP import *

AHPDDDD = np.array([
    [1, 1, 3, 3, 2, 3, 2],
    [1, 1, 3, 4, 2, 5, 1],
    [1 / 3, 1 / 3, 1, 4, 1 / 2, 3, 1 / 2],
    [1 / 3, 1 / 4, 1 / 4, 1, 1 / 3, 2, 1 / 2],
    [1 / 2, 1 / 2, 2, 3, 1, 3, 1],
    [1 / 3, 1 / 5, 1 / 3, 1 / 2, 1 / 3, 1, 1 / 3],
    [1 / 2, 1, 2, 2, 1, 3, 1]
])
AHPMissile = np.array([
    [1, 1, 2, 2, 1, 2, 2, 1],
    [1, 1, 3, 3, 2, 2, 2, 1],
    [1 / 2, 1 / 3, 1, 2, 1, 3, 1 / 2, 1 / 2],
    [1 / 2, 1 / 3, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1 / 3],
    [1, 1 / 2, 1, 2, 1, 2, 1, 1 / 2],
    [1 / 2, 1 / 2, 1 / 3, 1, 1 / 2, 1, 1 / 3, 1 / 3],
    [1 / 2, 1 / 2, 2, 2, 1, 3, 1, 1 / 2],
    [1, 1, 2, 3, 2, 3, 2, 1]
])
AHPAttackPlane = np.array([
    [1, 1 / 2, 3, 2, 3, 2, 2],
    [2, 1, 3, 2, 4, 3, 2],
    [1 / 3, 1 / 3, 1, 1 / 2, 2, 1 / 3, 1/3],
    [1 / 2, 1 / 2, 2, 1, 2, 2, 2],
    [1 / 3, 1 / 4, 1 / 2, 1 / 2, 1, 1 / 3, 1/2],
    [1 / 2, 1 / 3, 3, 1 / 2, 3, 1, 1/2],
    [1 / 2, 1 / 2, 3, 1 / 2, 2, 2, 1],
])
AHPSupportPlane = np.array([
    [1, 1 / 3, 2, 3, 2, 2],
    [3, 1, 3, 3, 2, 2],
    [1 / 2, 1 / 3, 1, 1, 1 / 2, 1 / 3],
    [1 / 3, 1 / 3, 1, 1, 1 / 2, 1 / 3],
    [1 / 2, 1 / 2, 2, 2, 1, 1 / 2],
    [1 / 2, 1 / 2, 3, 3, 2, 1],
])


# AHP 方法
def ahp(matrix):
    # 计算矩阵每列的乘积
    prod_matrix = np.prod(matrix, axis=1)
    # 计算每列的n次方根
    w_vector = prod_matrix ** (1 / len(matrix))
    # 归一化权重向量
    return w_vector / np.sum(w_vector)


# 熵权法
def entropy_weight(data: list):
    # 归一化处理
    p_matrix = data / np.sum(data, axis=0)

    # 计算熵值
    e = -np.sum(p_matrix * np.log(p_matrix), axis=0) / np.log(len(data))
    if np.sum(1 - e) == 0:
        return e / np.sum(e).tolist()
    else:
        return ((1 - e) / np.sum(1 - e)).tolist()


# 融合算法
def fusion_ahp_entropy(ahp_weights: list, entropy_weights: list, alpha=0.5):
    # 融合两种权重，alpha 是 AHP 权重的融合系数，(1-alpha) 是熵权法的融合系数
    return [x * alpha + y * (1 - alpha) for x, y in zip(ahp_weights, entropy_weights)]


class AHP_TA(object):
    """
    与墨子平台衔接的小波神经网络威胁评估算法
    """

    def __init__(self, targetData: list):
        # 各目标的所有可得信息，用于威胁评估、
        # DDDD：保卫要地重要性、毁伤概率、速度、射程、飞临时间、高度、突防能力
        # 其他DD：保卫要地重要性、毁伤概率、速度、RCS、飞临时间、高度、航路捷径、机动能力
        # 作战FJ：保卫要地重要性、飞机对地攻击作战效能、速度、RCS、高度、航路捷径
        # 作战支援FJ：保卫要地重要性、对地支援作战效能、速度、RCS、高度、距离
        super(AHP_TA, self).__init__()
        self.targetInfo = [[], [], [], []]  # 不同类型目标的威胁信息
        self.targetName = [[], [], [], []]  # 不同类型目标的名称
        self.targetsOrderedDict = None  # 所有目标的有序信息字典
        self.targetsOrderedType = None  # 所有目标的有序类型列表
        self.DataList = targetData
        self.index = 0

        # 随机一致性指标
        self.RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]

        self.AHPMatrixList = [AHPDDDD, AHPMissile, AHPAttackPlane, AHPSupportPlane]
        self.AHPWeights = []

        for AHPMatrix in self.AHPMatrixList:
            eig_values, eig_vectors = np.linalg.eig(AHPMatrix)
            # eigvalues为特征向量，eigvectors为特征值构成的对角矩阵（而且其他位置都为0，对角元素为特征值）
            max_index = np.argmax(eig_values)
            # argmax为获取最大特征值的下标,而且这里是获取实部
            max_eig = eig_values[max_index].real
            # 这里max_eig是最大的特征值
            eig_ = eig_vectors[:, max_index].real
            AHPWeight = (eig_ / eig_.sum()).tolist()

            n = AHPMatrix.shape[0]
            CR = 0
            if n > 15:
                print("无法判断一致性")
            else:
                CI = (max_eig - n) / (n - 1)
                if self.RI[n - 1] != 0:
                    CR = CI / self.RI[n]
                if CR < 0.1:
                    pass
                    # print("矩阵的一致性可以被接受")
                else:
                    print("矩阵的一致性不能被接受")
            self.AHPWeights.append(AHPWeight)

    def run(self):
        """
        处理目标的威胁评估信息
        :return:
        """

        self.targetsOrderedDict, self.targetsOrderedType = precessTargetsData(self.DataList)
        # 遍历self.targetsOrderedType，判断每个目标的类型，并将对应顺序的self.targetsOrderedDict中的目标信息提取出来，存入不同的列表中
        # self.targetsOrderedType：0-用于评估的数据，1-用于展示的数据，2-进攻要地，3-毁伤
        for name, target in self.targetsOrderedDict.items():
            if self.targetsOrderedType[self.index] == 0:  # DDDD
                self.targetName[0].append(name)
                self.targetInfo[0].append(target[0])
            elif self.targetsOrderedType[self.index] == 1:  # 导弹
                self.targetName[1].append(name)
                self.targetInfo[1].append(target[0])
            elif self.targetsOrderedType[self.index] == 2:  # 作战FJ
                self.targetName[2].append(name)
                self.targetInfo[2].append(target[0])
            elif self.targetsOrderedType[self.index] == 3:  # 作战支援FJ
                self.targetName[3].append(name)
                self.targetInfo[3].append(target[0])
            self.index += 1

        for typeIndex, targetInType in enumerate(self.targetInfo):
            threatInType = []
            if len(targetInType) > 0:
                ahpWeight = self.AHPWeights[typeIndex]

                targetInType = [target.tolist()[0] for target in targetInType]

                if len(targetInType) > 1:
                    # 计算熵权法权重
                    entropyWeight = entropy_weight(targetInType)
                    # 融合权重
                    fusedWeight = fusion_ahp_entropy(ahpWeight, entropyWeight)
                else:
                    fusedWeight = ahpWeight
                for target in targetInType:
                    threatInType.append(sum(a * b for a, b in zip(fusedWeight, target)))
                for i in range(len(threatInType)):
                    t = processTypeThreat(self.targetName[typeIndex][i], threatInType[i])
                    self.targetsOrderedDict[self.targetName[typeIndex][i]].append(t)
                    self.targetsOrderedDict[self.targetName[typeIndex][i]][1].append(round(t, 2))

        threat, damageList, baseList, info = [], [], [], []
        for name, target in self.targetsOrderedDict.items():
            info.append(target[1])
            baseList.append(target[2])
            damageList.append(target[3])
            threat.append(target[4])
        return threat, damageList, baseList

# example = [
#             ['超级眼镜蛇直升机 #2', 150.0, 609.6, 119.28175509951748, 37.9849475636907, 237.924423],
#            ['超级眼镜蛇直升机 #1', 150.0, 609.6, 119.35585515116178, 38.04372012621402, 237.916336],
#            ['超级眼镜蛇直升机 #4', 150.0, 609.6, 119.35542526538082, 38.04350788289485, 237.916138],
#            ['超级眼镜蛇直升机 #3', 150.0, 609.6, 119.43043898870889, 37.9848439385717, 237.95018],
#            ['超级眼镜蛇直升机 #5', 150.0, 609.6, 119.50461646558111, 37.92577833084922, 237.9838],
#            ['不明空中目标#90172', 550.0, 30.48, 119.10875557448992, 37.795617103121955, 274.336456],
#            ['不明空中目标#90173', 550.0, 30.48, 119.14644134542203, 37.79841379732579, 274.115448],
#            ['E-3G预警机 #1', 230.0, 10972.8, 119.59404212714817, 37.77859350306286, 274.424347],
#            ['RQ-180隐身无人侦察机 #1', 490.0, 10972.8, 118.80231120949514, 38.316254346517546, 278.128784],
#            ['S-2导弹[12万吨 核弹炸药] #1548', 6500.0, 5549.89063, 117.90892111511872, 37.56557964466659, 350.9159],
#            ['B-1B轰炸机 #4', 550.0, 93.44, 118.96054673741047, 37.68343324093701, 272.924347],
#            ['B-1B轰炸机 #6', 550.0, 30.48, 118.9936558159432, 37.52106929436113, 271.7967],
#            ['B-1B轰炸机 #5', 550.0, 30.48, 119.00957403894768, 37.528237586968686, 271.406921],
#            ['B-1B轰炸机 #3', 550.0, 30.48, 119.04053429114903, 37.680409633841656, 272.958649],
#            ]
# example = [['女武神无人机 #1', 480.0, 10972.8, 119.13320623571929, 37.85140950083703, 285.319427]]
# a = AHP_TA(example)
# a.run()
