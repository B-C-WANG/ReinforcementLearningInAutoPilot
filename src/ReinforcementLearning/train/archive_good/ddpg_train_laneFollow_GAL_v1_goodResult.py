# coding:utf-8
# Type: Private Author: BaoChuan Wang
'''

效果:并行训练效果不错,能够跑很长一段距离!最大跑了至少100000步
只在高速(Town04)环境中测试了

'''
import numpy as np
from ReinforcementLearning.Modules.Environments.Rewards import SafeDriveDistanceCost
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1

# 关于以下设置,请参考最初的A3c train waypoints GAL (global and local)
server_config = {
    "10.10.9.128": [2000],
}
n_workers_in_each_port = 5

spawn_index_for_each_car_in_worker = (0, 10, 20, 30, 40, 50, 60, 70, 80)
# 用随机数,扩大搜索
spawn_index_for_each_car_in_worker = np.random.randint(0, 100, size=100)

env_dict = {}
worker_kwargs = {}
model_kwargs = {}

for ip in server_config:
    for port in server_config[ip]:
        for i in range(n_workers_in_each_port):
            name = 'W_%s' % (str(ip) + "_" + str(port) + "_" + str(i))  # worker name
            env = LaneFollowEnv_v1(
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip=ip,
                carla_UE_port=port,
                vehicle_start_point_index=spawn_index_for_each_car_in_worker[i],
                wait_time_after_apply_action=0.1,
                # 这里是DDPG和A3C算法的区别,使用连续空间的action
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                # 对于DDPG不能使用有正值的reward,而使用cost
                reward_replace=SafeDriveDistanceCost(),
                drive_time_after_reset=2.0,
                kwargs_for_done_condition={
                    # 因为有预先开出,所以再次降速后直接done
                    "minimum_action_taken_on_low_velocity": 0
                }
            )
            env_dict[name] = env
            model_kwargs[name] = {
                # 预先求得准确q值给数据集,这里local模型和后面的global都需要!
                "use_pre_calculated_g": False,
                "gamma":0.9
            }
            worker_kwargs[name] = {
                 # 下面的variance和offset,第一个是油门,第二个是转向,具体查看相应action
                "start_variance_for_each_action": (1.0, 1.0), # 施加到模型输出action的方差
                "variance_decay_ratio_for_each_action": (0.995, 0.995),  # 上面施加的方差的衰减率
                "variance_decay_step": 100,  # 方差每隔多少步衰减
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
                      "use_pre_calculated_g":False,
                        "gamma":0.9
                  }).start()
