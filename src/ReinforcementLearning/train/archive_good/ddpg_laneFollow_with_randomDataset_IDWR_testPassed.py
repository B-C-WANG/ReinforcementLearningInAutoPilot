# coding:utf-8
# Type: Private Author: BaoChuan Wang
'''
随机的的数据集直接作用于RL模型中的state-action网络进行优化!

RL不进行优化,因为这个文件仅仅用于展示RL模型是否学习到了IL给出的数据集!

测试:无视state,直接给出1,0的action,很快模型给出的油门就到了0.7
而给出0,0的action,模型也能够维持在预测接近0的情况
直接给出0.4,0.3,然后训练频率加大,很快模型预测达到预定结果,有效!

'''
import random
import time
import numpy as np
from ReinforcementLearning.Modules.Environments.Rewards import \
    FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v1
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1
from ReinforcementLearning.ImitationLearning.ModelTrainers.ModelTrainers import FloatActionTrainer

# 关于以下设置,请参考最初的A3c train waypoints global and local
server_config = {
    "192.168.1.125": [2000],
}
n_workers_in_each_port = 1

spawn_index_for_each_car_in_worker = (0, 10, 20, 30, 40, 50, 60, 70, 80)
# 用随机数,扩大搜索
spawn_index_for_each_car_in_worker = np.random.randint(0, 100, size=100)

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
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip=ip,
                carla_UE_port=port,
                vehicle_start_point_index=spawn_index_for_each_car_in_worker[i],
                wait_time_after_apply_action=0.1,
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                reward_replace=FinalStepDriveDistanceAverageSpeedAverageDistanceToLaneCost_v1(),
                minimum_velocity=0.5,
                drive_time_after_reset=2.0,
                kwargs_for_done_condition={
                    "minimum_action_taken_on_low_velocity": 0
                }
            )
            env_dict[name] = env
            model_kwargs[name] = {
                "use_pre_calculated_g": use_pre_calculated_g,
                "gamma": gamma
            }
            worker_kwargs[name] = {
                "start_variance_for_each_action": (0, 0.0),
                "variance_decay_ratio_for_each_action": (0.995, 0.995),
                "variance_decay_step": 10,
                "start_offset_for_each_action": (0.0, 0.0),
                "offset_decay_value_for_each_action": (0.00, 0.00),
                "offset_decay_step": 2000
            }

agent = DDPG_Agent_GAL_v1(env_prototype_dict_for_workers=env_dict,
                          save_dir="./ddpg_ckpt/",
                          # 这两个参数是嵌套字典
                          kwargs_for_model_dict=model_kwargs,
                          kwargs_for_worker_dict=worker_kwargs,
                          kwargs_for_global_model={
                              # 预先求得准确q值给数据集
                              "use_pre_calculated_g": use_pre_calculated_g,
                              "gamma": gamma
                          })
model_trainer = FloatActionTrainer(
    # 输入state
    input_placeholder=agent.global_model.S,
    # 输出state的频率
    output_graph=agent.global_model.a,
    action_space_size=(agent.action_space,),
    tf_sess=agent.sess)
# 然后agent先启动
agent.start()

# 之后以一定频率每次给一个数据集训练,state随机,但是action只是油门1,转向0,预期学习到在任何情况下都是油门1,转向0
while 1:
    time.sleep(0.01)
    model_trainer.train(
        state_batch=np.random.random(size=(10, agent.observation_space)),
        # 全1和0的矩阵,注意打印出来看一下
        action_batch=np.stack([np.ones(shape=(10, 1)) * 0.4, np.ones(shape=(10, 1)) * 0.3], axis=1).reshape(10, 2))
