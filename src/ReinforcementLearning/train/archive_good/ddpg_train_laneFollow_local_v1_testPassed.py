# coding:utf-8
# Type: Private Author: BaoChuan Wang


'''
# 最新训练结果:效果非常好!能够走非常长的路而不出车道,转弯都很稳,不过训练时间长了会越开越慢,但是很难出车道线
'''



from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1
from ReinforcementLearning.Modules.Environments.Rewards import DistanceToLaneCost,SafeDriveDistanceCost

env = LaneFollowEnv_v1(
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip="10.10.9.128",
                carla_UE_port=2000,
                vehicle_start_point_index=0,
                wait_time_after_apply_action=0.1,
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                reward_replace=SafeDriveDistanceCost(),
                # 预先开车2.0秒
                drive_time_after_reset=2.0,
                kwargs_for_done_condition={
                    # 因为有预先开出,所以再次降速后直接done
                    "minimum_action_taken_on_low_velocity":0
                }
                #reward_replace=SafeDriveDistanceReward_v1(),


            )
# 注意使用的q值是手动计算得到的
DDPG_Agent_v1(env=env,kwargs_for_model={"gamma":0.9},use_model_with_pre_calculated_g=True).train()
