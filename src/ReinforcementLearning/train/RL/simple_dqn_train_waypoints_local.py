# coding:utf-8
# Type: Private Author: BaoChuan Wang

'''
描述:DQN过waypoint,效果较差
'''

from ReinforcementLearning.Modules.Environments.Environments_waypointTarget import CarlaConsecutiveWaypointsTargetEnv_v1
from ReinforcementLearning.Modules.Agents.DQN_Agent import DQNAgent_v1

env = CarlaConsecutiveWaypointsTargetEnv_v1(
                 carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                 carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                 carla_UE_ip="10.10.9.128",
                 carla_UE_port=2000,
)
DQNAgent_v1("ckpt/",env=env).train()

