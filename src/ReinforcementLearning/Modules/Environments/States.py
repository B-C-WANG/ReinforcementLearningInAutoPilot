# coding:utf-8
# Type: MultiAuthor

import math
import random
import numpy as np
from ReinforcementLearning.Modules.utils import MathUtils as myMath
from ClosedError import  ClosedError

class IState(object):

    def __init__(self, id):
        self.id = id

    def __str__(self):
        return self.id

    def merge_states(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def space_shape(self):
        raise NotImplementedError()


def concat_state_vectors(states):
    # 如果Env包含多个State,这里进行State给出的向量的合并
    return np.concatenate(states, axis=0)


class TrafficLightState_v1(IState):
    Author = "BaoChuan Wang"
    AllowImport = False
    '''
    输出:第一个为0,代表没有红灯(或者绿灯),第一个为1,代表红灯或者黄灯
    第二个为到交通灯的距离,超出100时为0
    否则为距离/100
    注意:实测为绿灯时,即使距离交通灯很近,distance也为None
    '''

    raise ClosedError("Code content is not open-source!")


class CenterLaneThreeTimesPolyFitParamState_v1(IState):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    # 车道线3次拟合参数state
    '''

    raise ClosedError("Code content is not open-source!")


class CarState_v2(IState):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    车辆的速度,加速度标量,和速度,加速度向量的xy,注意速度加速度都是相对于车辆坐标系
    '''

    raise ClosedError("Code content is not open-source!")


class MoveTargetAndCarState_v1(IState):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    用Target waypoint和车辆当前的速度等信息作为state
    '''



    raise ClosedError("Code content is not open-source!")

