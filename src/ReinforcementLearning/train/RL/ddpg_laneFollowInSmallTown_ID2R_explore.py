# coding: utf-8
# Type: Private Author: BaoChuan Wang


'''
描述:
在IL学习过后,将ckpt存储到ddpg_ckpt_IL,然后这里使用RL进行学习,不再进行IL
如果RL学习效果不好希望重新载入IL的权重,请复制ddpg_ckpt_IL_bak中的内容覆盖掉ddpg_ckpt_IL中的全部内容
注意因为有预先训练,所以探索的方差可以设置小一些
效果:

'''



from ReinforcementLearning.ImitationLearning.ModelTrainers.ModelTrainers import CarlaAgentOfLaneFollowFloatActionTrainer_v1
import random
import time
import numpy as np
from ReinforcementLearning.Modules.Environments.Rewards import SafeDriveDistanceCost
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1
from ReinforcementLearning.ImitationLearning.ModelTrainers.ModelTrainers import FloatActionTrainer

server_config = {
    "10.10.9.128": [2000],
}
# RL的agent只有3个
n_workers_in_each_port = 5


env_dict = {}
worker_kwargs = {}
model_kwargs = {}
carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg"
carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla"
use_pre_calculated_g = False
gamma = 0.99
for ip in server_config:
    for port in server_config[ip]:
        for i in range(n_workers_in_each_port):
            name = 'W_%s' % (str(ip) + "_" + str(port) + "_" + str(i))  # worker name
            env = LaneFollowEnv_v1(
                #'''
                # 有些地方道路曲率大(90度),车道线拟合不好导致认为出了车道,因此放宽车道条件
                # 但是放宽了条件后注意车辆会出现随意切换车道的现象!
                # 不过只要预先训练好,就根本不会出现变道的情况
                #'''
                maximum_distance_to_lane=0.5,
                # 建议只在debug阶段plot
                plot_lane_on_UE=False,
                use_random_start_point=True,
                carla_egg_path=carla_egg_path,
                carla_pythonAPI_path=carla_pythonAPI_path,
                carla_UE_ip=ip,
                carla_UE_port=port,
                wait_time_after_apply_action=0.1,
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                reward_replace=SafeDriveDistanceCost(),
                minimum_velocity=0.5,
                drive_time_after_reset=2.0,
                kwargs_for_done_condition={
                    "minimum_action_taken_on_low_velocity": 0
                }
            )
            env_dict[name] = env
            model_kwargs[name] = {
                "use_pre_calculated_g": use_pre_calculated_g,
                "gamma":gamma
            }
            worker_kwargs[name] = {
                "start_variance_for_each_action": (.3, .3),
                #"start_variance_for_each_action": (.0, .0),
                "variance_decay_ratio_for_each_action": (0.995, 0.995),
                "variance_decay_step": 10,
                "start_offset_for_each_action": (0.0, 0.0),
                "offset_decay_value_for_each_action": (0.00, 0.00),
                "offset_decay_step": 2000
            }

agent = DDPG_Agent_GAL_v1(env_prototype_dict_for_workers=env_dict,
                          # 这里使用IL的ckpt,直接从IL学习结果中继续学习!
                  save_dir="./ddpg_ckpt_IL/",
                  # 这两个参数是嵌套字典
                  kwargs_for_model_dict=model_kwargs,
                  kwargs_for_worker_dict=worker_kwargs,
                  kwargs_for_global_model={
                      # 预先求得准确q值给数据集
                      "use_pre_calculated_g":use_pre_calculated_g,
                        "gamma":gamma
                  })

agent.start()
