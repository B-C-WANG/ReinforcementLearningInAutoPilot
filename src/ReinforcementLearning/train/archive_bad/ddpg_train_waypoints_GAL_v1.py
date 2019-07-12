# coding:utf-8
# Type: Private Author: BaoChuan Wang



'''
FIXME ddpg 需要全为负的reward,waypoints环境暂时没有提供!

描述:
和local模型相同,训练得不到较好的结果

'''

import numpy as np
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_waypointTarget import CarlaConsecutiveWaypointsTargetEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1

# 关于以下设置,参考A3c train waypoints global and local
server_config = {

    "10.10.9.128": [2000],
}
n_workers_in_each_port = 1

spawn_index_for_each_car_in_worker = (0, 10, 20, 30, 40, 50, 60, 70, 80)
# 用随机数,扩大搜索
spawn_index_for_each_car_in_worker = np.random.randint(0, 100, size=100)

import tensorflow as tf

tf.random.set_random_seed(123)
np.random.seed(123)

env_dict = {}
worker_kwargs = {}
model_kwargs = {}
for ip in server_config:
    for port in server_config[ip]:
        for i in range(n_workers_in_each_port):
            name = 'W_%s' % (str(ip) + "_" + str(port) + "_" + str(i))  # worker name

            env = CarlaConsecutiveWaypointsTargetEnv_v1(
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip=ip,
                carla_UE_port=port,
                n_waypoint=100,
                # DDPG没有IL相对难训练,所以间隔小一些!
                waypoint_spacing=3,
                vehicle_start_point_index=spawn_index_for_each_car_in_worker[i],
                wait_time_after_apply_action=0.1,
                ratio_of_reaching=0.3,
                add_center_lane_state=True,
                # 这里是DDPG和A3C算法的区别,使用连续空间的action
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                # 实测DDPG和A3C表现差异很大,因此单独设计它的reward试试?
                #reward_replace=
            )
            env_dict[name] = env

            worker_kwargs[name] = {
                "start_variance": 0.0,# debug时方差小一些,便于观察走势
                "variance_decay": 0.99,
                "debug":True
            }
            # model_kwargs[name] = {
            # }

DDPG_Agent_GAL_v1(env_prototype_dict_for_workers=env_dict, save_dir="./a3c_gal_ckpt/",
                  kwargs_for_worker_dict=worker_kwargs,
                  ).start()
