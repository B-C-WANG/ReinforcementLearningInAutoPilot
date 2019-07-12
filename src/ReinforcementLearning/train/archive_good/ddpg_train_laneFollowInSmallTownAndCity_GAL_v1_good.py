# coding:utf-8
# Type: Private Author: BaoChuan Wang
'''


描述:
在城镇环境(Town02)和城市环境(Town05)中完成纯RL学习,使其做到较好的车道跟随

效果:
稳定,车速偏慢,过90°弯道和路口较顺畅
有速度控制:平常加速,速度过高会减速,转弯前会减速

测试FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v2()有效

训练过程中现象:
   1 一开始会有持续减速现象(低于速度done)
   2 然后会逐渐变化到一直左转或者右转(然后出车道线done)
   3 然后左转右转幅度减小,能够纠正回来,不断纠正使得车辆左右摇晃,在左右摇晃中前行(在此种状态下能够完成转弯)
   因为上述摇晃的现象实际上是因为为了探索而给转向加了噪声造成的,因此当噪声衰减后,车辆控制会逐渐稳定
   4 在3过后逐渐优化,可能出现2~3一直循环的现象,可能是因为路口的bug引起的
   然后会出现一开始加速到很高(因为有速度的正reward),然后在转弯处快速减速过弯的情况



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
                maximum_distance_to_lane=0.3,
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
