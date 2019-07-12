# coding:utf-8
import numpy as np
import time
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.DataAnalysisTools.DataAnalysis import DataCollector

import tensorflow as tf
from ClosedError import ClosedError
class IModelTrainers(object):
    '''
    用于RL中state-action部分的注入训练
    给出state-action对,然后用新的优化器直接优化RL中的state-action局部网络
    '''
    pass

    def __init__(self,
                 # 输入state
                 input_placeholder,
                 # 从输入state到预测action的计算图
                 output_graph,
                 action_space_size,
                 *args,
                 **kwargs):
        self.input_placeholder = input_placeholder
        self.output_placeholder = tf.placeholder(shape=(None,)+action_space_size, dtype=tf.float32)
        self.output_graph = output_graph

class CarlaAgentOfLaneFollowFloatActionTrainer_v1(IModelTrainers):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    描述
    采用carla内置的Agent来提供训练的数据集,因为carla内置的agent效果很好!
    需要使用CarlaEngine,和env的实现方式有相近之处!
    
    state默认使用lane follower的默认值,也就是车辆速度xy + 车辆加速度xy + 车道4个参数
    action是油门/刹车 和转向
    '''
    @staticmethod
    def test_agent():
        trainer = CarlaAgentOfLaneFollowFloatActionTrainer_v1(debug_for_agent=True)
        while 1:
            trainer.step()

    def __init__(self,
                 # 为了兼容之前的代码增加的replace(这里应该是将env作为无默认值的参数)
                 env_replace=None,
                 # 给env加的参数,dict
                 kwargs_for_env=None,
                 input_placeholder=None,
                 output_graph=None,
                 # 注意优化的sess要和RL的模型在同一个sess,否则不会修改到权重
                 tf_sess=None,
                 action_space_size=None,
                 lr=0.00001,
                 # 给agent action添加的噪声量,便于RL学习纠正行为,噪声不会进入训练集,只会反映在控制中
                 variance_for_each_action=(0.,0.),
                 # 只测试carla的agent,不建立tf计算图和优化!
                 debug_for_agent=False,
                 # 当操作一定步数过后,重新生成agent来控制车辆!主要是避免agent一直卡住,同时切换位置使数据集范围更广
                 steps_for_carla_agent_reset=1200,
                 # 是否无视信号灯
                 ignore_traffic_light=True,
                 # 数据收集
                 data_collector=None,):
        raise ClosedError("Code content is not open-source!")

    def train(self, state_batch, action_batch):
        '''
        :param state_batch: state数据集,shape需要为None, + state shape, 如(1,7)
        :param action_batch: action数据集,shape需要为None, + action shape, 如(1,2)
        :return:
        '''
        raise ClosedError("Code content is not open-source!")

    def reset(mySelf):
        '''
        这个方法相当于重载了env中的reset
        本类的self变成mySelf
        而env就变为self,这样里面的代码接近env自己的reset方法,便于识别
        因为python中函数是对象,所以直接复制实现重载
        '''
        raise ClosedError("Code content is not open-source!")

    def override_agent_is_light_red(self,*args,**kwargs):
        '''
        重载的红绿灯检测方法,直接不要红绿灯检测
        这个方法重载的是RoamingAgent源代码中,基类Agent里面的方法
        '''
        return (False,None)

    def step(mySelf):
        # 这里重载env中的step方法,注意原来step中必要的内容要保留
        raise ClosedError("Code content is not open-source!")


class FloatActionTrainer(IModelTrainers):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    直接给出数据集进行训练
    '''
    def __init__(self,
                 input_placeholder,
                 output_graph,
                 # 注意优化的sess要和RL的模型在同一个sess,否则不会修改到权重
                 tf_sess,
                 action_space_size,
                 lr=0.001,
                 *args,
                 **kwargs):

        super(FloatActionTrainer, self).__init__(input_placeholder,
                                                 output_graph,
                                                 action_space_size)
        self.sess = tf_sess
        self.loss = tf.losses.mean_squared_error(self.output_graph,self.output_placeholder)
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self,state_batch,action_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.input_placeholder:state_batch,
            self.output_placeholder:action_batch
        })

if __name__ == '__main__':
    CarlaAgentOfLaneFollowFloatActionTrainer_v1.test_agent()
