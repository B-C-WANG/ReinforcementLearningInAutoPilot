# coding:utf-8
# Type: MultiAuthor
import random
from ClosedError import ClosedError

class IAction(object):
    def __init__(self, id, *args, **kwargs):
        self.id = id

    def __str__(self):
        return self.id

    def reset(self):
        return

    @property
    def space_shape(self):
        # 提供action的向量大小
        raise NotImplementedError()


class ContinuousSteeringVelocityBrakeAction_v1(IAction):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    浮点数action，油门和转向和刹车
    '''
    raise ClosedError("Code content is not open-source!")


class EqualPriorProbaDistributionSimpleSteeringAction_v1(IAction):
    Author = "BaoChuan Wang"
    AllowImport = False
    '''
    设置好相等先验概率分布的action能够有助于训练的加速!
    
    等先验概率分布的action
    SimpleSteeringAction_v1的问题在于,大部分action都应该是0,也就是加速,action的先验分布不平均
    现在设置分布均匀的action,通过模型后求得state下的条件概率
    '''
    raise ClosedError("Code content is not open-source!")


class HighPrecisionControl_v1(IAction):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    高精度控制
    '''
    raise ClosedError("Code content is not open-source!")


class SimpleSteeringAction_v1(IAction):
    Author = "BaoChuan Wang"
    AllowImport = True

    raise ClosedError("Code content is not open-source!")

