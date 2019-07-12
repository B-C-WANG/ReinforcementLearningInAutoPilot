# coding:utf-8


'''
FIXME ddpg 需要全为负的reward,waypoints环境暂时没有提供!
# https://blog.csdn.net/qq_30615903/article/details/80776715
# ddpg的好处是直接采用浮点数的输出!

ddpg结合waypoint的训练有很大问题,需要debug

'''



from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_v1
from ReinforcementLearning.Modules.Environments.Environments_waypointTarget import CarlaConsecutiveWaypointsTargetEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1

env = CarlaConsecutiveWaypointsTargetEnv_v1(
    carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
    carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
    carla_UE_ip="10.10.9.128",
    carla_UE_port=2000,
    n_waypoint=200,
    waypoint_spacing=3,
    ratio_of_reaching=0.2,
    add_center_lane_state=True,
    action_replace=ContinuousSteeringVelocityBrakeAction_v1()
)
DDPG_Agent_v1(env=env).train()
