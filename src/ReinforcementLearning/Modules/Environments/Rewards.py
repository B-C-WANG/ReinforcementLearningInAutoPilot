# coding: utf-8
# Type: MultiAuthor

import numpy as np
import math
import warnings
from ClosedError import ClosedError

class IReward(object):
    '''
    cost是特殊的reward,cost只能为负,像DDPG这样的算法,不允许cost为正,否则loss为负,越优化loss绝对值越大

    '''

    def __init__(self, id):
        # name用于存储识别使用
        self.id = id

    def get_reward(self, *args, **kwargs):
        raise NotImplementedError()
        # return float

    def __str__(self):
        return self.id

    def reset(self):
        # 某些reward需要reset
        pass


class CarToPathDistanceDeltaReward_v1(IReward):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    适用于Env_waypointsTarget,用前后的变化量作为rewards的计算依据
    分别是
    action前后车辆到车道线垂直距离的delta
    action前后车辆方向角和way方向角差的delta
    action前后速度的delta
    
    重要!如果采用这种方法,需要确保数值是连续的,因为这里的假设条件是上面用于计算delta的指标都是连续的
     这样才能反映action对delta的影响,如果数值不完全对action连续,则会出现action外的情况影响rewards
     常见的就是waypoints随着车辆的变动而变动
     如果这种变动是增益的,或者幅度小的,可以忽略,
     具体的例子:车辆前进,waypoints更新后(waypoints总是车辆前面5m的点,每隔2.5m更新),夹角突然变小,如果转弯垂直距离会突然变大
    '''

    raise ClosedError("Code content is not open-source!")


class FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v2(IReward):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    适用于Env_laneFollow
    和v1的区别是加大了速度和车道线的贡献,v1是稳定训练的,但是最后车速很慢
    '''

    raise ClosedError("Code content is not open-source!")


class FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v1(IReward):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    适用于Env_laneFollow
    最后一步给出安全行驶距离+,平均速度+,以及平均到车道线的距离-的总和cost
    '''

    raise ClosedError("Code content is not open-source!")


class SafeDriveDistanceSpeedAndDistanceToLaneCost(IReward):
    Author = "BaoChuan Wang"
    AllowImport = True

    '''
    最后一步加上总距离cost,其他情况下使用到车道线距离和速度的cost
    '''

    raise ClosedError("Code content is not open-source!")


class SafeDriveDistanceCost(IReward):
    Author = "BaoChuan Wang"
    AllowImport = True
    Archive = True
    '''
    在max distance行驶了多少相对距离作为cost

    每一步加上速度,然后只在最后一步done的时候给出总reward,其他状态给出0的reward
    用它减去max distance,使其变为负,然后除以max distance,注意超过max distance会有问题
    '''

    raise ClosedError("Code content is not open-source!")


class DistanceToLaneCost(IReward):
    Author = "BaoChuan Wang"
    AllowImport = False
    '''
    车道线距离作为cost
    '''

    def __init__(self, ):
        super(DistanceToLaneCost, self).__init__("")

    def get_reward(self, distance_to_center_lane, velocity, done, *args, **kwargs):
        reward = -abs(distance_to_center_lane)
        # FIXME:单独给一个Cost基类判断reward大小
        if reward > 0.0:
            raise ValueError("Cost can not >0.0")
        return reward


class VelocityDistanceToLaneReward(IReward):
    Author = "BaoChuan Wang"
    AllowImport = False
    """
    任何时候每一步给出速度+和到车道线距离-的reward
    """

    raise ClosedError("Code content is not open-source!")


class SafeDriveDistanceReward_v1(IReward):
    '''
    每一步加上速度,然后只在最后一步done的时候给出总reward,其他状态给出0的reward
    '''

    raise ClosedError("Code content is not open-source!")


class SafeDriveStepsVelocityAndDistanceToCenterLaneReward_v1(IReward):
    '''
    每一步使用速度+,和距离车道线距离-为reward
    每一步积累速度的reward,然后只在最后一步done的时候给出总reward,当然也要加上速度和车道线距离reward分量

    '''

    raise ClosedError("Code content is not open-source!")


class OnlyCarToTargetWaypointDistanceRatioReward_v1(IReward):
    Author = "BaoChuan Wang"
    AllowImport = False
    '''
    没有必要设置太多的reward参数,让模型自己去学习,
    比如车到pathway线段的垂直距离,可以不用设置,因为能够达到waypoint终点的不会太过离谱
    另外在is done中可以设置越快达到终点,reward越高!
    '''

    raise ClosedError("Code content is not open-source!")


class CarToPathAndTargetWaypointDistanceReward_v1(IReward):
    Author = "BaoChuan Wang"
    AllowImport = False
    '''
    用车辆到pathway线段的垂直距离
    增加车辆到终点距离
    注意:对于速度的reward不要太高,否则模型可能一直选择加速
    '''

    raise ClosedError("Code content is not open-source!")

