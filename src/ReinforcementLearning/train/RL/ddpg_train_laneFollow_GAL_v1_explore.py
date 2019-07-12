# coding:utf-8
# Type: Private Author: BaoChuan Wang
'''



TODO: 分别完成过红绿灯，跟车，避障等环境，最后合并到一起

TODO: 探索使得车辆跑得更好方法!


测试1:
只在最后一步,用路程作为最后的reward,不预先计算q,gamma较大以至于计算q时会出现从后往前的发散
但实际上训练较好(NOTICE 也就是gamma向前传播不收敛不会出现问题!)

测试2:
只在最后一步,用平均速度+,平均和车道线距离-,以及路程+作为最后的reward,不预先计算q
结果:
不错,直线车辆摇摆有一定收敛性,可以过270度的转弯,但是速度偏慢!

测试3:
在测试2中使用的reward上改进,增加速度和车道线距离的权重,减少路程权重
结果车速较快,但是之后摇摆幅度大,对于急转弯是很难学习到的!


reward设计:
    还是采取终点reward设计原则,尽可能只在最后一步给出reward,其他部分reward都是0
    但是比如过红绿灯这种,就是过了为0,不过为-1

    只要保证
    reward不为正就可以训练成功,否则会出现训练结束后车辆一直左转等固定action的情况

    q值发散似乎不会影响训练!
    q值发散:手动gamma向前传播的时候,发散是0,1,1,1,5变成9,8,7,6,5而不是0.2,1.3,2.4,3.6,5这样

    模型中的use_pre_calculated_g可要可不要,实测可能为True时训练更快,但是可能有过拟合问题

'''
import random
import numpy as np
from ReinforcementLearning.Modules.Environments.Rewards import \
    FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v1,SafeDriveDistanceCost,\
    FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v2
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1


# 关于以下设置,请参考最初的A3c train waypoints global and local
server_config = {
    "10.10.9.128": [2000],
}
n_workers_in_each_port = 5

env_dict = {}
worker_kwargs = {}
model_kwargs = {}

use_pre_calculated_g = False
gamma = 0.99
for ip in server_config:
    for port in server_config[ip]:
        for i in range(n_workers_in_each_port):
            name = 'W_%s' % (str(ip) + "_" + str(port) + "_" + str(i))  # worker name
            env = LaneFollowEnv_v1(
                # 在城镇环境中,车道线约束放宽,否则在路口有异常(即使设置很大导致车辆可以随意变道)
                maximum_distance_to_lane=0.5,
                # 随机起点,有助于测试过拟合情况
                use_random_start_point=True,
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip=ip,
                carla_UE_port=port,
                wait_time_after_apply_action=0.1,
                # 这里是DDPG和A3C算法的区别,使用连续空间的action
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                # 对于DDPG不能使用有正值的reward,而使用cost

                # 有效,很慢:使用安全驾驶的距离作为cost,只有最后一步有cost,然后按照gamma向前传播
                #reward_replace=SafeDriveDistanceCost(),

                # 使用到车道线距离作为cost
                #reward_replace=DistanceToLaneCost(),

                # 有效,但很慢:使用速度,车道线,总路程综合reward
                #reward_replace=FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v1(),
                # 有效,速度较快
                reward_replace=FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v2(),
                # 最小速度降低,便于转弯
                minimum_velocity=0.5,
                drive_time_after_reset=2.0,
                kwargs_for_done_condition={
                    # 因为有预先开出,所以再次降速后直接done
                    "minimum_action_taken_on_low_velocity": 0
                }
            )
            env_dict[name] = env
            model_kwargs[name] = {
                # 预先求得准确q值给数据集,这里local模型和后面的global都需要!
                "use_pre_calculated_g": use_pre_calculated_g,
                "gamma":gamma
            }
            worker_kwargs[name] = {
                 # 下面的variance和offset,第一个是油门,第二个是转向,具体查看相应action
                "start_variance_for_each_action": (1, 1.0), # 施加到模型输出action的方差
                "variance_decay_ratio_for_each_action": (0.995, 0.995),  # 上面施加的方差的衰减率
                "variance_decay_step": 10,  # 方差每隔多少步衰减
                # 这里的油门的修正+0.5
                "start_offset_for_each_action": (0.0, 0.0),
                # 油门修正的衰减,等于0时一直保持修正值
                "offset_decay_value_for_each_action": (0.00, 0.00),
                "offset_decay_step": 2000
            }

DDPG_Agent_GAL_v1(env_prototype_dict_for_workers=env_dict,
                  save_dir="./ddpg_ckpt/",
                  # 这两个参数是嵌套字典
                  kwargs_for_model_dict=model_kwargs,
                  kwargs_for_worker_dict=worker_kwargs,
                  kwargs_for_global_model={
                      # 预先求得准确q值给数据集
                      "use_pre_calculated_g":use_pre_calculated_g,
                        "gamma":gamma
                  }).start()
