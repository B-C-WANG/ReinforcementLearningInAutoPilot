# coding: utf-8
# Type: Private Author: BaoChuan Wang


'''

学习carla自带的Agent的行为!在高速环境中!

注意把carla地图设置成Town04等高速环境

效果:
效果极佳!
RL的agent很快学习到了carla agent的行为,在直线段开得很稳
在弯道区域也能够非常稳定的行驶!
建议在复杂环境中进行模仿学习

注意:这种学习是十分局限的,因为RL没有自己去探索,而是纯粹模仿,因此对于边界条件的控制不好,
 RL还需要多探索!

 注意:carla agent绘制的车道线和车辆行驶路线存在不一致的情况!对于城市场景会有较大影响

测试1:
即使是添加了一开始探索的方差,RL也能够学习得很好!


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
    "192.168.1.125": [2000],
}
# RL的agent只有3个
n_workers_in_each_port = 3


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
                "start_variance_for_each_action": (1, 1.0),
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
                      "use_pre_calculated_g":use_pre_calculated_g,
                        "gamma":gamma
                  })
carla_model_trainer = CarlaAgentOfLaneFollowFloatActionTrainer_v1(
    # 学习率降低,因为只是引导作用
    lr=0.0001,
    # 输入state
    input_placeholder=agent.global_model.S,
    # 输出state的频率
    output_graph=agent.global_model.a,
    action_space_size=(agent.action_space,),
    tf_sess=agent.sess,
    kwargs_for_env={
        'carla_egg_path':carla_egg_path,
        'carla_pythonAPI_path':carla_pythonAPI_path,
            # 随意选择一个ip和port
        'carla_UE_ip':list(server_config.keys())[0],
        'carla_UE_port':server_config[list(server_config.keys())[0]][0],
    }
)
# 然后agent先启动
agent.start()

# 之后以一定频率每次给一个数据集训练,state随机,但是action只是油门1,转向0,预期学习到在任何情况下都是油门1,转向0
while 1:
    # carla train里面内置训练和频率控制函数
    carla_model_trainer.step()
