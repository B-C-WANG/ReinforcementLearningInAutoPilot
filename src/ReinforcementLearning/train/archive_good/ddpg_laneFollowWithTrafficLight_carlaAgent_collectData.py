# coding: utf-8
# Type: Private Author: BaoChuan Wang


'''
描述:
在车道跟随和红绿灯的state下,运行carla的agent,收集数据到文件
之后可以通过决策树等模型对收集的数据训练,根据决策树的可视化结果判断模型的合理性和边界
'''
from ReinforcementLearning.ImitationLearning.ModelTrainers.ModelTrainers import \
    CarlaAgentOfLaneFollowFloatActionTrainer_v1
from ReinforcementLearning.Modules.Environments.Rewards import SafeDriveDistanceCost
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow_withTrafficLight import \
    LaneFollowEnv_WithTrafficLight_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1
from ReinforcementLearning.Modules.DataAnalysisTools.DataAnalysis import DataCollector

print("Test Only on Town02! Change distance_regard_as_brake_light_rules when use other Town!")

data_collector = DataCollector(save_dir="./laneFollowTrafficLightData/",
                               data_name_list=["state","action"],
                               data_length_when_save=2000)
carla_model_trainer = CarlaAgentOfLaneFollowFloatActionTrainer_v1(
    env_replace=LaneFollowEnv_WithTrafficLight_v1(
        # 打印红绿灯信息debug
        do_traffic_light_debug=False,
        # 不进行闯红灯判断,因为carla agent几乎不会闯红灯
        distance_regard_as_brake_light_rules=-9999,
        maximum_distance_to_lane=0.8,
        use_random_start_point=True,
        carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
        carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
        # 随意选择一个ip和port
        carla_UE_ip="192.168.1.125",
        carla_UE_port=2000,
        wait_time_after_apply_action=0.1,
        action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
        reward_replace=SafeDriveDistanceCost(), ),
    # 给agent添加一定噪声,便于学习纠正行为
    variance_for_each_action=(.5, .5),
    # 不进行训练,收集数据
    debug_for_agent=True,
    # 开启红绿灯
    ignore_traffic_light=False)


while 1:

    state,action=    carla_model_trainer.step()
    data_collector.add_data([state,action])
