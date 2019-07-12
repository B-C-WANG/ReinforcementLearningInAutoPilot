# coding:utf-8
from ClosedError import ClosedError
class IDone(object):
    '''
    done用于给出训练结束的标志
    '''
    def is_done(self,*args,**kwargs):
        done = 0
        reward = 0
        return done,reward

    def reset(self,*args,**kwargs):
        pass

class SoftLowVelocityFarFromLaneThroughRedLightDone(IDone):
    '''
    结束的条件:
    1.除了在等红绿灯且为红灯,其他任何时候,连续一定步数速度都低于某个值
    2.任何时候,距离车道线太远
    3.闯红灯(距离面前的status为红的灯的车前进方向的距离小于某个值,这个值需要设置好!)

    '''
    raise ClosedError("Code content is not open-source!")


class LowVelAndFarFromLaneDoneAfterNSteps_v1(IDone):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    结束的条件:
    1.至少进行某个step后低于某个速度done
    2.距离车道线太远done(使用车道线的c0参数即可)
    '''
    raise ClosedError("Code content is not open-source!")


class DistanceToTargetLowestStepAndLowVelDone_v1(IDone):
    Author = "BaoChuan Wang"
    AllowImport = False
    '''
    比起一般的done的True和False,这个条件匹配于Env_waypointTarget
    done为True时,还需要分成功和失败,因为成功是设置下一个路径点,失败是重新训练
    1.至少进行某个step后低速done,视为fail
    2.超过路径点而没有到达路径点done,fail
    2.到达路径点done,success
    done过后修改reward
    '''
    raise ClosedError("Code content is not open-source!")


