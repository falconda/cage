################################################################################
# 威胁评估部分飞机目标的数据预处理模块
################################################################################
import math
import numpy as np

# DDDD的射程
DDDDTargetRange = {
    "“流星-3”型中程弹道导弹[常规弹头]": 1296,
    "“北极星-2”潜射中程弹道导弹弹头": 2748,
    "S-2导弹[12万吨 核弹炸药]": 3055.8
}

# DDDD的突防能力
DDDDTargetPenetration = {
    "“流星-3”型中程弹道导弹[常规弹头]": "较弱",
    "“北极星-2”潜射中程弹道导弹弹头": "中",
    "S-2导弹[12万吨 核弹炸药]": "中"
}

# DD的射程
MissileTargetRange = {
    "战斧-3000亚声速反舰/对陆导弹": 2500,
    'AGM-114N型“地狱火II”空对地导弹[温压弹]': 9.26,
    "吸气式高超声速导弹": 555.6,
    "GBU-38(V)1/B联合直接攻击炸弹": 22.224,
    "GBU-53/B 小直径炸弹II": 111.12,
    "AGM-65K型“小牛”空地战术导弹": 14.816,
    "AGM-65G2型“小牛”空地战术导弹": 14.816,
    "“流星-3”型中程弹道导弹[常规弹头]": 1296,
    "巡飞弹": 5,
    "无人机抛射弹": 87,
    "“北极星-2”潜射中程弹道导弹弹头": 2748,
    "S-2导弹[12万吨 核弹炸药]": 3055.8
}

# DD的DP值，代替毁伤
MissileTargetDP = {
    "战斧-3000亚声速反舰/对陆导弹": 454,
    'AGM-114N型“地狱火II”空对地导弹[温压弹]': 12.8,
    "吸气式高超声速导弹": 50000,
    "GBU-38(V)1/B联合直接攻击炸弹": 130.5,
    "GBU-53/B 小直径炸弹II": 72,
    "AGM-65K型“小牛”空地战术导弹": 61.2,
    "AGM-65G2型“小牛”空地战术导弹": 61.2,
    "“流星-3”型中程弹道导弹[常规弹头]": 1200,
    "巡飞弹": 45,
    "无人机抛射弹": 87,
    "“北极星-2”潜射中程弹道导弹弹头": 10000,
    "S-2导弹[12万吨 核弹炸药]": 120000
}

# DD的RCS值
MissileTargetRCS = {
    "战斧-3000亚声速反舰/对陆导弹": 0.22,
    'AGM-114N型“地狱火II”空对地导弹[温压弹]': 0.010,
    "吸气式高超声速导弹": 0.3,
    "GBU-38(V)1/B联合直接攻击炸弹": 0.038,
    "GBU-53/B 小直径炸弹II": 0.021,
    "AGM-65K型“小牛”空地战术导弹": 0.018,
    "AGM-65G2型“小牛”空地战术导弹": 0.018,
    "巡飞弹": 0.055,
    "无人机抛射弹": 0.023,
}

# DD的机动能力
MissileTargetAction = {
    "战斧-3000亚声速反舰/对陆导弹": 0.8,
    'AGM-114N型“地狱火II”空对地导弹[温压弹]': 0.4,
    "吸气式高超声速导弹": 1,
    "GBU-38(V)1/B联合直接攻击炸弹": 0.6,
    "GBU-53/B 小直径炸弹II": 0.6,
    "AGM-65K型“小牛”空地战术导弹": 0.4,
    "AGM-65G2型“小牛”空地战术导弹": 0.4,
}

basePosition = {
    "基地1": (117.64555555, 37.8666),
    "基地2": (117.87194444, 37.7155555),
    "基地3": (117.64527777, 37.543333)
}

baseBlood = {
    "基地1": 2500,
    "基地2": 2500,
    "基地3": 2500,
}

# FJ的RCS值
PlaneTargetRCS = {
    "EC-130H干扰机": 39.8,
    "E-3G预警机": 75.9,
    "女武神无人机": 2.3E-05,
    "诡骗丽影无人战斗机": 0.00021,
    "U-2S侦察机": 17.4,
    "F-16DJ战斗机": 1.6,
    "B-1B轰炸机": 18.2,
    "B-52H轰炸机": 107.2,
    "RQ-180隐身无人侦察机": 5.9E-05,
    "F-15E战斗轰炸机": 5.8,
    "超级眼镜蛇直升机": 1.3,
    "大黄蜂战斗机": 4.2,
    "枪骑兵轰炸机": 18.2
}

# FJ的对地作战攻击效能
PlaneAttack = {
    "EC-130H干扰机": {"最大航程": 3944.76, "翼展": 40.4, "全长": 28.9, "RCS": 39.8, "最大使用过载": 4, "最低突防高度": 600, "最高突防速度": 600,
                   "武器射程": 0, "武器数量": 0, "机上外挂点": 9, "电子对抗能力系数": 1.2, "导航能力系数": 1.03, "发现目标能力系数": 1.02, "装甲系数": 1,
                   "修正系数": 0, "武器精度系数": 0, "最大载弹量": 0
                   },
    "E-3G预警机": {"最大航程": 5315.24, "翼展": 44.4, "全长": 46.6, "RCS": 75.9, "最大使用过载": 4, "最低突防高度": 100, "最高突防速度": 963.4,
                "武器射程": 0, "武器数量": 0, "机上外挂点": 0, "电子对抗能力系数": 0.9, "导航能力系数": 1.1, "发现目标能力系数": 1.5, "装甲系数": 0.9,
                "修正系数": 0, "武器精度系数": 0, "最大载弹量": 0
                },
    "女武神无人机": {"最大航程": 3941, "翼展": 6.7, "全长": 8.8, "RCS": 0.012, "最大使用过载": 4, "最低突防高度": 100, "最高突防速度": 1111.2,
               "武器射程": 22.224, "武器数量": 5, "机上外挂点": 0, "电子对抗能力系数": 0.8, "导航能力系数": 0.8, "发现目标能力系数": 0.9, "装甲系数": 0.8,
               "修正系数": 0.7, "武器精度系数": 0.9, "最大载弹量": 1.25
               },
    "诡骗丽影无人战斗机": {"最大航程": 3889.2, "翼展": 18.92, "全长": 18.92, "RCS": 0.0012, "最大使用过载": 4, "最低突防高度": 100,
                  "最高突防速度": 1101.94, "武器射程": 111.12, "武器数量": 8, "机上外挂点": 0, "电子对抗能力系数": 0.8, "导航能力系数": 0.8,
                  "发现目标能力系数": 0.9, "装甲系数": 0.8,
                  "修正系数": 0.7, "武器精度系数": 0.9, "最大载弹量": 1.25
                  },
    "F-16DJ战斗机": {"最大航程": 3000, "翼展": 9.5, "全长": 14.5, "RCS": 1.6, "最大使用过载": 8, "最低突防高度": 100, "最高突防速度": 1713.1,
                  "武器射程": 14.816, "武器数量": 6, "机上外挂点": 9, "电子对抗能力系数": 1.10, "导航能力系数": 1.03, "发现目标能力系数": 1.02, "装甲系数": 1,
                  "修正系数": 1, "武器精度系数": 1, "最大载弹量": 8
                  },
    "B-1B轰炸机": {"最大航程": 5000.4, "翼展": 53.1, "全长": 51.7, "RCS": 106.2, "最大使用过载": 4, "最低突防高度": 100, "最高突防速度": 916.74,
                "武器射程": 2500, "武器数量": 16, "机上外挂点": 6, "电子对抗能力系数": 0.7, "导航能力系数": 0.75, "发现目标能力系数": 0.7, "装甲系数": 0.8,
                "修正系数": 1, "武器精度系数": 1, "最大载弹量": 60
                },
    "B-52H轰炸机": {"最大航程": 8000, "翼展": 56.4, "全长": 48.5, "RCS": 107.2, "最大使用过载": 3.5, "最低突防高度": 75, "最高突防速度": 944.52,
                 "武器射程": 555.6, "武器数量": 1, "机上外挂点": 2, "电子对抗能力系数": 0.8, "导航能力系数": 0.8, "发现目标能力系数": 0.8, "装甲系数": 0.9,
                 "修正系数": 1, "武器精度系数": 1.5, "最大载弹量": 31.751
                 },
    "RQ-180隐身无人侦察机": {"最大航程": 8334, "翼展": 40, "全长": 10, "RCS": 0.01, "最大使用过载": 4, "最低突防高度": 60, "最高突防速度": 907.48,
                      "武器射程": 0, "武器数量": 0, "机上外挂点": 0, "电子对抗能力系数": 0.6, "导航能力系数": 0.9, "发现目标能力系数": 1.3, "装甲系数": 0.5,
                      "修正系数": 0, "武器精度系数": 0, "最大载弹量": 0
                      },
    "F-15E战斗轰炸机": {"最大航程": 5550, "翼展": 13.1, "全长": 19.3, "RCS": 5.8, "最大使用过载": 4, "最低突防高度": 100, "最高突防速度": 1713.1,
                   "武器射程": 14.816, "武器数量": 12, "机上外挂点": 11, "电子对抗能力系数": 0.9, "导航能力系数": 1.03, "发现目标能力系数": 0.9, "装甲系数": 1,
                   "修正系数": 1, "武器精度系数": 1, "最大载弹量": 7.3
                   },
    "超级眼镜蛇直升机": {"最大航程": 3944.76, "翼展": 4.3, "全长": 13.9, "RCS": 1.3, "最大使用过载": 5, "最低突防高度": 60, "最高突防速度": 333.36,
                 "武器射程": 9.26, "武器数量": 40, "机上外挂点": 4, "电子对抗能力系数": 0.8, "导航能力系数": 0.8, "发现目标能力系数": 0.6, "装甲系数": 0.8,
                 "修正系数": 0.7, "武器精度系数": 0.5, "最大载弹量": 2.615
                 },
    "大黄蜂战斗机": {"最大航程": 2346, "翼展": 11.4, "全长": 17.1, "RCS": 4.2, "最大使用过载": 7.6, "最低突防高度": 91, "最高突防速度": 1713.1,
               "武器射程": 14.816, "武器数量": 10, "机上外挂点": 11, "电子对抗能力系数": 0.9, "导航能力系数": 1.0, "发现目标能力系数": 1.1, "装甲系数": 1,
               "修正系数": 0.7, "武器精度系数": 1, "最大载弹量": 6.2
               },
    "枪骑兵轰炸机": {"最大航程": 11999, "翼展": 41.7, "全长": 47.8, "RCS": 18.2, "最大使用过载": 4.5, "最低突防高度": 60, "最高突防速度": 1296.4,
               "武器射程": 22.224, "武器数量": 24, "机上外挂点": 6, "电子对抗能力系数": 0.8, "导航能力系数": 0.85, "发现目标能力系数": 0.83, "装甲系数": 1,
               "修正系数": 1, "武器精度系数": 1, "最大载弹量": 60
               },
    "F22": {"最大航程": 2500, "翼展": 13.6, "全长": 18.9, "RCS": 0.007, "最大使用过载": 9.5, "最低突防高度": 50, "最高突防速度": 1203.8,
            "武器射程": 0, "武器数量": 0, "机上外挂点": 0, "电子对抗能力系数": 1.2, "导航能力系数": 1, "发现目标能力系数": 1, "装甲系数": 1,
            "修正系数": 1, "武器精度系数": 1, "最大载弹量": 1
            },
    "F35": {"最大航程": 2000, "翼展": 10.7, "全长": 15.4, "RCS": 0.01, "最大使用过载": 8, "最低突防高度": 50, "最高突防速度": 1203.8,
            "武器射程": 22.224, "武器数量": 2, "机上外挂点": 0, "电子对抗能力系数": 1, "导航能力系数": 1, "发现目标能力系数": 1, "装甲系数": 1,
            "修正系数": 1, "武器精度系数": 1, "最大载弹量": 5.089
            }

}

# FJ的对地作战支援效能
PlaneSupport = {
    "EC-130H干扰机": {"最大航程": 3944.76, "翼展": 40.4, "全长": 28.9, "RCS": 39.8, "最高飞行速度": 601.9, "升限": 9753.6, "通信能力系数": 1,
                   "探测距离": 55, "干扰距离": 230, "跟踪目标数": 0, "引导目标数": 0, "侦察能力系数": 0.1, "电子对抗能力系数": 1, "续航": 3
                   },
    "E-3G预警机": {"最大航程": 7400, "翼展": 44.4, "全长": 46.6, "RCS": 75.9, "最高飞行速度": 963.04, "升限": 12192, "通信能力系数": 1,
                "探测距离": 648.2, "干扰距离": 0, "跟踪目标数": 100, "引导目标数": 50, "侦察能力系数": 0, "电子对抗能力系数": 0.6, "续航": 3
                },
    "EA-18G“咆哮者": {"最大航程": 3254, "翼展": 18.2, "全长": 16.2, "RCS": 6, "最高飞行速度": 1713.1, "升限": 13716, "通信能力系数": 1,
                   "探测距离": 55, "干扰距离": 160, "跟踪目标数": 0, "引导目标数": 0, "侦察能力系数": 0.3, "电子对抗能力系数": 0.8, "续航": 5
                   },
    "RQ-180隐身无人侦察机": {"最大航程": 20000, "翼展": 40, "全长": 10, "RCS": 1, "最高飞行速度": 907.48, "升限": 15240, "通信能力系数": 0.6,
                      "探测距离": 111.12, "干扰距离": 0, "跟踪目标数": 0, "引导目标数": 0, "侦察能力系数": 1, "电子对抗能力系数": 0.4, "续航": 24
                      },
    "U-2S侦察机": {"最大航程": 8000, "翼展": 31.4, "全长": 19.2, "RCS": 39.8, "最高飞行速度": 796.36, "升限": 25908, "通信能力系数": 0.4,
                "探测距离": 0, "干扰距离": 0, "跟踪目标数": 0, "引导目标数": 0, "侦察能力系数": 0.9, "电子对抗能力系数": 0.9, "续航": 8
                },
}


# 保卫要地的重要性
def processBaseImportance(base: str):
    if base == "基地1":
        return 0.85
    elif base == "基地2":
        return 0.80
    elif base == "基地3":
        return 0.90


# 航路捷径
def processCut(cut: float):
    if cut < 0:
        return 0  # 远离
    elif cut <= 1:
        return 1
    elif cut <= 5:
        return 1 - 0.3 * (cut - 1) / (5 - 1)
    elif cut <= 10:
        return 0.7 - 0.4 * (cut - 5) / (10 - 5)
    else:
        return 0.3


################################################################################
# 威胁评估DDDD目标的数据预处理模块
################################################################################
# 1.DD攻击的要地重要性

# 2.DD的毁伤概率
def processDDDDDamage(type: str, base: str):
    dp = MissileTargetDP[type] / baseBlood[base]
    return dp if dp < 1 else 1


# 3.速度
def processDDDDSpeed(MissileSpeed: float, MissileHeight: float):
    soundSpeed = 340 - 3.9 * MissileHeight / 1000
    speedMa = MissileSpeed / 3.6 / soundSpeed
    if speedMa <= 1:
        return 0.2, speedMa
    elif speedMa <= 5:
        return 0.2 + 0.1 * (speedMa - 1) / (5 - 1), speedMa
    elif speedMa <= 10:
        return 0.3 + 0.3 * (speedMa - 5) / (10 - 5), speedMa
    elif speedMa <= 20:
        return 0.6 + 0.4 * (speedMa - 10) / (20 - 10), speedMa
    else:
        return 1, speedMa


# 4.射程
def processDDDDRange(Range: float):
    if Range <= 1000:
        return 0.2
    elif Range <= 5000:
        return 0.2 + 0.3 * (Range - 1000) / (5000 - 1000)
    elif Range <= 8000:
        return 0.5 + 0.5 * (Range - 5000) / (8000 - 5000)
    else:
        return 1


# 5.飞临时间
def processDDDDTime(time: float):
    if time <= 50:
        return 1
    elif time <= 100:
        return 1 - 0.3 * (time - 50) / (100 - 50)
    elif time <= 300:
        return 0.7 - 0.3 * (time - 100) / (300 - 100)
    elif time <= 500:
        return 0.4 - 0.2 * (time - 300) / (500 - 300)
    else:
        return 0.2


# 6.高度
def processDDDDHeight(height: float):
    if height <= 10000:
        return 1
    elif height <= 40000:
        return 1 - 0.4 * (height - 10000) / (40000 - 10000)
    elif height <= 200000:
        return 0.6 - 0.4 * (height - 40000) / (200000 - 40000)
    else:
        return 0.2


# 7.突防能力
def processDDDDPenetration(p: str):
    if p is "弱":
        return 0.2
    elif p is "较弱":
        return 0.4
    elif p is "中":
        return 0.6
    elif p is "较强":
        return 0.8
    else:
        return 1


################################################################################
# 威胁评估其他DD目标的数据预处理模块
################################################################################
# 1.DD攻击的要地重要性

# 2.DD的毁伤概率
def processMissileDamage(type: str, base: str):
    dp = MissileTargetDP[type] / baseBlood[base]
    return dp if dp < 1 else 1


# 3.速度
def processMissileSpeed(MissileSpeed: float, MissileHeight: float):
    soundSpeed = 340 - 3.9 * MissileHeight / 1000
    speedMa = MissileSpeed / 3.6 / soundSpeed
    if speedMa <= 0.8:
        return 0.2, speedMa
    elif speedMa <= 1.2:
        return 0.2 + 0.3 * (speedMa - 0.8) / (1.2 - 0.8), speedMa
    elif speedMa <= 5:
        return 0.5 + 0.2 * (speedMa - 1.2) / (5 - 1.2), speedMa
    elif speedMa <= 10:
        return 0.7 + 0.3 * (speedMa - 5) / (10 - 5), speedMa
    else:
        return 1, speedMa


# 4.RCS
def processMissileRCS(RCS: float):
    if RCS <= 0.01:
        return 1
    elif RCS <= 0.1:
        return 1 - 0.3 * (RCS - 0.01) / (0.1 - 0.01)
    elif RCS <= 0.2:
        return 0.7 - 0.4 * (RCS - 0.1) / (0.2 - 0.1)
    else:
        return 0.3


# 5.飞临时间
def processMissileTime(time: float):
    if time <= 50:
        return 1
    elif time <= 100:
        return 1 - 0.3 * (time - 50) / (100 - 50)
    elif time <= 300:
        return 0.7 - 0.3 * (time - 100) / (300 - 100)
    elif time <= 500:
        return 0.4 - 0.2 * (time - 300) / (500 - 300)
    else:
        return 0.2


# 6.高度
def processMissileHeight(height: float):
    if height <= 100:
        return 1
    elif height <= 1000:
        return 1 - 0.3 * (height - 100) / (1000 - 100)
    elif height <= 20000:
        return 0.7 - 0.5 * (height - 1000) / (20000 - 1000)
    else:
        return 0.2


# 7.航路捷径

# 8.机动能力
def processMissileAction(action: str):
    if action is "弱":
        return 0.2
    elif action is "较弱":
        return 0.4
    elif action is "中":
        return 0.6
    elif action is "较强":
        return 0.8
    else:
        return 1


################################################################################
# 威胁评估作战FJ目标的数据预处理模块
################################################################################
# 1.作战FJ攻击的要地的重要性

# 2.作战FJ的对地支援作战效能
def processPlaneDamage(planeType: str):
    attackDict = PlaneAttack[planeType]
    C6 = ((10 / attackDict["翼展"]) * (15 / attackDict["全长"]) * (5 / attackDict["RCS"])) ** 0.0625
    C2 = 0.35 * C6 + 0.05 * attackDict["装甲系数"] + 0.15 * (attackDict["最大使用过载"] / 9) + 0.2 * (
            100 / attackDict["最低突防高度"]) + 0.25 * (attackDict["最高突防速度"] / 1200)
    C5 = (attackDict["武器射程"] / 3) * attackDict["修正系数"] * np.sqrt(attackDict["武器数量"]) + 1
    C4 = 0.2 * attackDict["机上外挂点"] / 15 + 0.5 * attackDict["武器精度系数"] + 0.3 * attackDict["发现目标能力系数"]
    C1 = attackDict["电子对抗能力系数"]
    C3 = attackDict["导航能力系数"]
    Rmax = attackDict["最大航程"]  # 最大航程
    Wmax = attackDict["最大载弹量"]  # 最大载弹量

    limit = 1e-2
    groundAttack = C1 * (10 * np.log(C2) + 10 * np.log(C3) + 10 * np.log(C4) + np.log(Rmax) + np.log(C5) + (
        np.log(Wmax) if Wmax > limit else -1))

    if groundAttack <= 4:
        return 0.2
    elif groundAttack <= 7:
        return 0.2 + 0.3 * (groundAttack - 4) / (7 - 4)
    elif groundAttack <= 10:
        return 0.5 + 0.2 * (groundAttack - 7) / (10 - 7)
    elif groundAttack <= 12:
        return 0.7 + 0.3 * (groundAttack - 10) / (12 - 10)
    else:
        return 1


# 3.速度
def processPlaneSpeed(PlaneSpeed: float, PlaneHeight: float):
    soundSpeed = 340 - 3.9 * PlaneHeight / 1000
    speedMa = PlaneSpeed / 3.6 / soundSpeed
    if speedMa <= 0.3:
        return 0.4, speedMa
    elif speedMa <= 0.8:
        return 0.4 + 0.3 * (speedMa - 0.3) / (0.8 - 0.3), speedMa
    elif speedMa <= 1.5:
        return 0.7 + 0.3 * (speedMa - 0.8) / (1.5 - 0.8), speedMa
    else:
        return 1, speedMa


# 4.RCS
def processPlaneRCS(RCS: str):
    RCS = float(RCS)
    if RCS <= 0.01:
        return 1
    elif RCS <= 0.1:
        return 1 - 0.3 * (RCS - 0.01) / (0.1 - 0.01)
    elif RCS <= 1:
        return 0.7 - 0.2 * (RCS - 0.1) / (1.0 - 0.1)
    elif RCS <= 10:
        return 0.5 - 0.3 * (RCS - 1) / (10.0 - 1.0)
    else:
        return 0.2


# 5.高度
def processPlaneHeight(height: float):
    if height <= 100:
        return 1
    elif height <= 1000:
        return 1 - 0.2 * (height - 100) / (1000 - 100)
    elif height <= 7000:
        return 0.8 - 0.2 * (height - 1000) / (7000 - 1000)
    elif height <= 15000:
        return 0.6 - 0.3 * (height - 7000) / (15000 - 7000)
    else:
        return 0.3


# 等距方位投影将经纬度转换为平面坐标
def projection(lon: float, lat: float):
    R = 6356.9088
    lon0, lat0 = 114, 36
    x = R * (math.cos(lon0) * math.sin(lon) - math.sin(lon0) * math.cos(lon) * math.cos(lat)) \
        / (math.sin(lon0) * math.sin(lon) + math.cos(lon0) * math.cos(lon) * math.cos(lat))
    y = R * math.cos(lon) * math.sin(lat) \
        / (math.sin(lon0) * math.sin(lon) + math.cos(lon0) * math.cos(lon) * math.cos(lat))
    return x, y


# 6.航路捷径

################################################################################
# 威胁评估作战支援FJ目标的数据预处理模块
################################################################################
# 1.作战FJ攻击的要地的重要性

# 2.作战支援飞机的对地支援作战效能
def processPlaneSupport(planeType: str):
    attackDict = PlaneSupport[planeType]

    C6 = ((10 / attackDict["翼展"]) * (15 / attackDict["全长"]) * (5 / attackDict["RCS"])) ** 0.0625
    C2 = 0.3 * C6 + 0.4 * attackDict["升限"] / 8000 + 0.3 * (attackDict["最高飞行速度"] / 600)
    C3 = 0.2 * attackDict["探测距离"] / 320 + 0.2 * (attackDict["跟踪目标数"] / 100 + attackDict["引导目标数"] / 100) + 0.4 * \
         attackDict["干扰距离"] / 320 + 0.2 * attackDict["侦察能力系数"]
    C4 = attackDict["通信能力系数"]
    C1 = attackDict["电子对抗能力系数"]
    Rmax = attackDict["最大航程"]  # 最大航程
    Wmax = attackDict["续航"]  # 最大载弹量
    limit = 1e-2
    groundSupport = C1 * (np.log(C2) + 1 * np.log(C3) + 1 * np.log(C4)) + np.log(Rmax) + np.log(
        Wmax)
    if groundSupport <= 4:
        return 0.2
    elif groundSupport <= 7:
        return 0.2 + 0.3 * (groundSupport - 4) / (7 - 4)
    elif groundSupport <= 10:
        return 0.5 + 0.2 * (groundSupport - 7) / (10 - 7)
    elif groundSupport <= 12:
        return 0.7 + 0.3 * (groundSupport - 10) / (12 - 10)
    else:
        return 1


# 3.速度
def processSupportPlaneSpeed(PlaneSpeed: float, PlaneHeight: float):
    soundSpeed = 340 - 3.9 * PlaneHeight / 1000
    speedMa = PlaneSpeed / 3.6 / soundSpeed
    if speedMa <= 0.3:
        return 0.4, speedMa
    elif speedMa <= 0.8:
        return 0.4 + 0.3 * (speedMa - 0.3) / (0.8 - 0.3), speedMa
    elif speedMa <= 1.5:
        return 0.7 + 0.3 * (speedMa - 0.8) / (1.5 - 0.8), speedMa
    else:
        return 1, speedMa


# 4.RCS
def processSupportPlaneRCS(RCS: str):
    RCS = float(RCS)
    if RCS <= 1:
        return 1
    elif RCS <= 10:
        return 1 - 0.3 * (RCS - 1) / (10 - 1)
    elif RCS <= 100:
        return 0.7 - 0.5 * (RCS - 10) / (100 - 10)
    else:
        return 0.2


# 5.高度
def processSupportPlaneHeight(height: float):
    if height <= 1000:
        return 0.2
    elif height <= 8000:
        return 0.2 + 0.3 * (height - 1000) / (8000 - 1000)
    elif height <= 10000:
        return 0.5 + 0.3 * (height - 8000) / (10000 - 8000)
    elif height <= 15000:
        return 0.8 + 0.2 * (height - 10000) / (15000 - 10000)
    else:
        return 1


# 6.距离
def processSupportPlaneDis(dis: float):
    if dis <= 10:
        return 1
    elif dis <= 100:
        return 1 - 0.3 * (dis - 1000) / (8000 - 1000)
    elif dis <= 300:
        return 0.7 - 0.4 * (dis - 8000) / (10000 - 8000)
    else:
        return 0.3


def processSingleDDDDData(missile: list):
    """
    DDDD：保卫要地重要度、毁伤概率、速度、射程、飞临时间、高度、突防能力
    :param missile: list[0名称，1速度，2高度，3经度，4纬度，5航向角]
    :return: 量化后的数据,1行7列的ndArray数组
    """
    missileResult = np.zeros((1, 7))
    missileInfo = []
    baseCut = []
    baseDis = []
    missileType = missile[0].split("#", 1)[0][:-1]
    for base, position in basePosition.items():
        dx = abs((missile[3] - position[0]) * 111 * math.cos(math.radians(position[1])))
        dy = abs((missile[4] - position[1]) * 111)
        dis = math.sqrt(dx ** 2 + dy ** 2)
        a = math.atan2(dy, dx)
        angle = a / math.pi * 180
        cut = 0.0
        if 180 < missile[5] < 270:
            cut = dis * math.sin(math.radians(abs(angle - (270 - missile[5]))))
        elif missile[5] > 270:
            cut = dis * math.sin(math.radians(abs(angle - (missile[5] - 270))))
        else:
            cut = -1
        baseDis.append(dis)
        baseCut.append(cut)
    minValue = min(baseCut)
    minValueIndex = baseCut.index(minValue)
    time = float(baseDis[minValueIndex]) / float(missile[1]) * 3600
    missileResult[0][0] = processBaseImportance("基地" + str(minValueIndex + 1))  # 保卫要地重要性
    missileResult[0][1] = processDDDDDamage(missileType, "基地" + str(minValueIndex + 1))  # 毁伤概率，以DP表示
    missileResult[0][2], speedMa = processDDDDSpeed(missile[1], missile[2])  # 速度：马赫数
    missileResult[0][3] = processDDDDRange(DDDDTargetRange[missileType])  # 射程
    missileResult[0][4] = processDDDDTime(time)  # 飞临时间
    missileResult[0][5] = processDDDDHeight(missile[2])  # 高度
    missileResult[0][6] = processDDDDPenetration(DDDDTargetPenetration[missileType])  # 突防能力

    # missileInfo.append(missileType)  # 0目标类型
    # missileInfo.append("基地" + str(minValueIndex + 1))  # 1进攻基地
    # missileInfo.append(round(dp, 4) if dp < 0.01 else round(dp, 2))  # 2毁伤概率
    # missileInfo.append(round(speedMa, 2))  # 3速度
    # missileInfo.append(rcs if rcs < 0.01 else round(rcs, 2))  # 4RCS
    # missileInfo.append(time if time < 0.01 else round(time, 2))  # 5飞临时间
    # missileInfo.append(missile[2])  # 6高度
    # missileInfo.append(minValue if minValue < 0.01 else round(minValue, 2))  # 7航路捷径
    # missileInfo.append(MissileTargetAction[missileType])  # 8机动能力

    return missileResult, missileInfo, missileResult[0][1], minValueIndex


def processSingleMissileData(missile: list):
    """
    导弹：保卫要地重要度、毁伤概率、速度、RCS、飞临时间、高度、航路捷径、机动能力
    :param missile: list[0名称，1速度，2高度，3经度，4纬度，5航向角]
    :return: 量化后的数据,1行9列的ndArray数组
    """
    missileResult = np.zeros((1, 8))
    baseCut = []
    baseDis = []
    missileInfo = []
    missileType = missile[0].split("#", 1)[0][:-1]
    dp = MissileTargetDP[missileType]
    rcs = MissileTargetRCS[missileType]

    for base, position in basePosition.items():
        dx = abs((missile[3] - position[0]) * 111 * math.cos(math.radians(position[1])))
        dy = abs((missile[4] - position[1]) * 111)
        dis = math.sqrt(dx ** 2 + dy ** 2)
        a = math.atan2(dy, dx)
        angle = a / math.pi * 180
        cut = 0.0
        if 180 < missile[5] < 270:
            cut = dis * math.sin(math.radians(abs(angle - (270 - missile[5]))))
        elif missile[5] > 270:
            cut = dis * math.sin(math.radians(abs(angle - (missile[5] - 270))))
        else:
            cut = -1
        baseDis.append(dis)
        baseCut.append(cut)
    minValue = min(baseCut)
    minValueIndex = baseCut.index(minValue)
    missileResult[0][7] = processCut(minValue)  # 航路捷径

    time = float(baseDis[minValueIndex]) / float(missile[1]) * 3600
    missileResult[0][0] = processBaseImportance("基地" + str(minValueIndex + 1))  # 保卫要地重要性
    missileResult[0][1] = processMissileDamage(missileType, "基地" + str(minValueIndex + 1))  # 毁伤概率，以DP计算
    missileResult[0][2], speedMa = processMissileSpeed(missile[1], missile[2])  # 马赫数
    missileResult[0][3] = processMissileRCS(rcs)  # RCS
    missileResult[0][4] = processMissileTime(time)  # 飞临时间
    missileResult[0][5] = processPlaneHeight(missile[2])  # 高度
    missileResult[0][6] = MissileTargetAction[missileType]  # 机动能力

    # missileInfo.append(missileType)  # 0目标类型
    # missileInfo.append("基地" + str(minValueIndex + 1))  # 1进攻基地
    # missileInfo.append(round(dp, 4) if dp < 0.01 else round(dp, 2))  # 2毁伤概率
    # missileInfo.append(round(speedMa, 2))  # 3速度
    # missileInfo.append(rcs if rcs < 0.01 else round(rcs, 2))  # 4RCS
    # missileInfo.append(time if time < 0.01 else round(time, 2))  # 5飞临时间
    # missileInfo.append(missile[2])  # 6高度
    # missileInfo.append(minValue if minValue < 0.01 else round(minValue, 2))  # 7航路捷径
    # missileInfo.append(MissileTargetAction[missileType])  # 8机动能力

    return missileResult, missileInfo, missileResult[0][1], minValueIndex


def processSinglePlaneData(plane: list):
    """
    处理单个飞机目标的信息
    飞机：  目标类型、保卫要地重要性、飞机对地攻击效能、速度、RCS、高度、航路捷径
    :param plane: list[0名称，1速度，2高度，3经度，4纬度，5航向角]
    """
    planeResult = np.zeros((1, 6))
    planeInfo = []
    baseCut = []
    planeType = plane[0].split("#", 1)[0][:-1]  # 目标类型

    for base, position in basePosition.items():
        dx = abs((plane[3] - position[0]) * 111 * math.cos(math.radians(position[1])))
        dy = abs((plane[4] - position[1]) * 111)
        dis = math.sqrt(dx ** 2 + dy ** 2)
        a = math.atan2(dy, dx)
        angle = a / math.pi * 180
        cut = 0.0
        if 180 < plane[5] < 270:
            cut = dis * math.sin(math.radians(abs(angle - (270 - plane[5]))))
        elif plane[5] > 270:
            cut = dis * math.sin(math.radians(abs(angle - (plane[5] - 270))))
        else:
            cut = -1
        baseCut.append(cut)
    minValue = min(baseCut)
    minValueIndex = baseCut.index(minValue)
    planeResult[0][0] = processBaseImportance("基地" + str(minValueIndex + 1))  # 保卫要地重要性
    planeResult[0][1] = processPlaneDamage(planeType)  # 作战效能
    planeResult[0][2], speedMa = processPlaneSpeed(plane[1], plane[2])  # 速度
    planeResult[0][3] = processPlaneRCS(PlaneTargetRCS[planeType])  # RCS
    planeResult[0][4] = processPlaneHeight(plane[2])  # 高度
    planeResult[0][5] = processCut(minValue)  # 航路捷径

    # planeInfo.append(planeType)  # 0目标类型
    # planeInfo.append("基地" + str(minValueIndex + 1))  # 1进攻基地
    # planeInfo.append(processPlaneDamage(planeType))  # 2作战效能
    # planeInfo.append(round(speedMa, 2))  # 3速度
    # planeInfo.append(
    #     PlaneTargetRCS[planeType] if PlaneTargetRCS[planeType] < 0.01 else round(PlaneTargetRCS[planeType], 2))  # 4RCS
    # planeInfo.append(plane[2])  # 5高度
    # planeInfo.append(minValue if minValue < 0.01 else round(minValue, 2))  # 6航路捷径

    return planeResult, planeInfo, planeResult[0][1], minValueIndex


def processSingleSupportPlaneData(plane: list):
    """
    处理单个飞机目标的信息
    飞机：  目标类型、保卫要地重要性、飞机对地攻击效能、速度、RCS、高度、航路捷径
    :param plane: list[0名称，1速度，2高度，3经度，4纬度，5航向角]
    """
    planeResult = np.zeros((1, 6))
    planeInfo = []
    baseDis = []
    baseCut = []
    planeType = plane[0].split("#", 1)[0][:-1]  # 目标类型

    for base, position in basePosition.items():
        dx = abs((plane[3] - position[0]) * 111 * math.cos(math.radians(position[1])))
        dy = abs((plane[4] - position[1]) * 111)
        dis = math.sqrt(dx ** 2 + dy ** 2)
        a = math.atan2(dy, dx)
        angle = a / math.pi * 180
        cut = 0.0
        if 180 < plane[5] < 270:
            cut = dis * math.sin(math.radians(abs(angle - (270 - plane[5]))))
        elif plane[5] > 270:
            cut = dis * math.sin(math.radians(abs(angle - (plane[5] - 270))))
        else:
            cut = -1
        baseDis.append(dis)
        baseCut.append(cut)
    minValue = min(baseCut)
    dis = min(baseDis)
    minValueIndex = baseCut.index(minValue)
    planeResult[0][0] = processBaseImportance("基地" + str(minValueIndex + 1))  # 保卫要地重要性
    planeResult[0][1] = processPlaneSupport(planeType)  # 作战效能
    planeResult[0][2], speedMa = processSupportPlaneSpeed(plane[1], plane[2])  # 速度
    planeResult[0][3] = processSupportPlaneRCS(PlaneTargetRCS[planeType])  # RCS
    planeResult[0][4] = processSupportPlaneHeight(plane[2])  # 高度
    planeResult[0][5] = processSupportPlaneDis(dis)  # 距离

    # planeInfo.append(planeType)  # 0目标类型
    # planeInfo.append("基地" + str(minValueIndex + 1))  # 1进攻基地
    # planeInfo.append(processPlaneDamage(planeType))  # 2作战效能
    # planeInfo.append(round(speedMa, 2))  # 3速度
    # planeInfo.append(
    #     PlaneTargetRCS[planeType] if PlaneTargetRCS[planeType] < 0.01 else round(PlaneTargetRCS[planeType], 2))  # 4RCS
    # planeInfo.append(plane[2])  # 5高度
    # planeInfo.append(minValue if minValue < 0.01 else round(minValue, 2))  # 6航路捷径

    return planeResult, planeInfo, planeResult[0][1], minValueIndex


def processWtaData(target):
    targetResult = np.zeros((1, 6), dtype=object)
    planeType = target.strName.split("#", 1)[0][:-1]  # 目标类型
    targetResult[0][0] = target.strName  # 目标名称

    # 目标类型
    if "战斗机" in planeType:
        targetResult[0][1] = 0
    elif "轰炸机" in planeType:
        targetResult[0][1] = 1
    elif "无人机" in planeType:
        targetResult[0][1] = 2
    elif "预警机" in planeType or "干扰机" in planeType or "侦察机" in planeType:
        targetResult[0][1] = 3
    elif "弹" in planeType:
        targetResult[0][1] = 4
    else:
        targetResult[0][1] = -1  # 不明目标

    targetResult[0][2] = [target.dLongitude, target.dLatitude] # 经度纬度
    targetResult[0][3] = target.fCurrentSpeed  # 速度
    targetResult[0][4] = target.fCurrentHeading  # 方位角

    # 作战范围
    if "弹道导弹" in planeType or "核弹" in planeType:
        targetResult[0][5] = DDDDTargetRange[planeType]
    elif ("导弹" in planeType and planeType.find("导弹") > 0) or "炸弹" in planeType:
        targetResult[0][5] = MissileTargetRange[planeType]
    elif "战斗机" in planeType or "直升机" in planeType or "轰炸机" in planeType or "女武神无人机" in planeType or "诡骗丽影无人战斗机" in planeType:
        targetResult[0][5] = PlaneAttack[planeType]["武器射程"]
    elif "干扰机" in planeType:
        targetResult[0][5] = PlaneSupport[planeType]["干扰距离"]
    elif "侦察机" in planeType or "预警机" in planeType:
        targetResult[0][5] = PlaneSupport[planeType]["探测距离"]
    else:
        targetResult[0][5] = 0

    return targetResult


def compute_damage(damage_list_0, damage_list_1):
    fac_name_0 = damage_list_0[0]
    fac_name_1 = damage_list_1[0]
    fac_damage_1 = damage_list_1[1]
    value_all = 0
    damage = 0

    JD = 4
    WQ = 1
    LD = 2

    for item in fac_name_0:
        if item in fac_name_1:
            if "基地" in item:
                damage += fac_damage_1[fac_name_1.index(item)] * JD
            elif "HQ" in item:
                damage += fac_damage_1[fac_name_1.index(item)] * WQ
            elif "LD" in item:
                damage += fac_damage_1[fac_name_1.index(item)] * LD
        elif item not in fac_name_1:
            if "基地" in item:
                damage += JD
            elif "HQ" in item:
                damage += WQ
            elif "LD" in item:
                damage += LD

        if "基地" in item:
            value_all += JD
        elif "HQ" in item:
            value_all += WQ
        elif "雷达" in item:
            value_all += LD


    result = damage/value_all

    return result

