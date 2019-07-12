# coding: utf-8
# Type: Private Author: BaoChuan Wang


'''
描述:
纯粹模仿学习效果测试!
加红绿灯行为
在Town02测试!
不同地图的闯红灯设置距离不同,因此需要固定地图

效果:
有效,能够在红绿等下停止,绿灯时也能够启动,
'''
from ReinforcementLearning.ImitationLearning.ModelTrainers.ModelTrainers import \
    CarlaAgentOfLaneFollowFloatActionTrainer_v1
from ReinforcementLearning.Modules.Environments.Rewards import SafeDriveDistanceCost
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow_withTrafficLight import \
    LaneFollowEnv_WithTrafficLight_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1

print("Test Only on Town02! Change distance_regard_as_brake_light_rules when use other Town!")
server_config = {
    "192.168.1.125": [2000],
}
# RL的agent只有3个,还有一个IL agent
n_workers_in_each_port = 3

env_dict = {}
worker_kwargs = {}
model_kwargs = {}
carla_egg_path = "/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg"
carla_pythonAPI_path = "/home/wang/Desktop/carla/PythonAPI/carla"
use_pre_calculated_g = False
gamma = 0.99
for ip in server_config:
    for port in server_config[ip]:
        for i in range(n_workers_in_each_port):
            name = 'W_%s' % (str(ip) + "_" + str(port) + "_" + str(i))  # worker name
            env = LaneFollowEnv_WithTrafficLight_v1(
                # 闯红灯距离是观察carla agent停车时距离红灯距离得到的
                distance_regard_as_brake_light_rules=3.0,
                # 放宽车道条件
                maximum_distance_to_lane=0.8,
                use_random_start_point=True,
                carla_egg_path=carla_egg_path,
                carla_pythonAPI_path=carla_pythonAPI_path,
                carla_UE_ip=ip,
                carla_UE_port=port,
                wait_time_after_apply_action=0.1,
                action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
                reward_replace=SafeDriveDistanceCost(), )
            env_dict[name] = env
            model_kwargs[name] = {
                "use_pre_calculated_g": use_pre_calculated_g,
                "gamma": gamma
            }
            worker_kwargs[name] = {
                # 因为模仿,所以不进行学习
                "do_RL_learn": False,
                "start_variance_for_each_action": (.0, .0),
                "variance_decay_ratio_for_each_action": (0.995, 0.995),
                "variance_decay_step": 10,
                "start_offset_for_each_action": (0.0, 0.0),
                "offset_decay_value_for_each_action": (0.00, 0.00),
                "offset_decay_step": 2000
            }

agent = DDPG_Agent_GAL_v1(env_prototype_dict_for_workers=env_dict,
                          save_dir="./ddpg_ckpt_IL_traffic_light/",
                          # 这两个参数是嵌套字典
                          kwargs_for_model_dict=model_kwargs,
                          kwargs_for_worker_dict=worker_kwargs,
                          kwargs_for_global_model={
                              # 预先求得准确q值给数据集
                              "use_pre_calculated_g": use_pre_calculated_g,
                              "gamma": gamma
                          })
carla_model_trainer = CarlaAgentOfLaneFollowFloatActionTrainer_v1(
    env_replace=LaneFollowEnv_WithTrafficLight_v1(
        # 打印红绿灯信息debug
        do_traffic_light_debug=False,
        # 不进行闯红灯判断,因为carla agent几乎不会闯红灯
        distance_regard_as_brake_light_rules=-9999,
        maximum_distance_to_lane=0.8,
        use_random_start_point=True,
        carla_egg_path=carla_egg_path,
        carla_pythonAPI_path=carla_pythonAPI_path,
        # 随意选择一个ip和port
        carla_UE_ip=list(server_config.keys())[0],
        carla_UE_port=server_config[list(server_config.keys())[0]][0],
        wait_time_after_apply_action=0.1,
        action_replace=ContinuousSteeringVelocityBrakeAction_v1(),
        reward_replace=SafeDriveDistanceCost(), ),
    # 引导RL学习,需要将lr调低,主要模仿则调高一些
    lr=0.001,
    # 给agent添加一定噪声,便于学习纠正行为
    variance_for_each_action=(.5, .5),
    # 输入state
    input_placeholder=agent.global_model.S,
    # 输出state的频率
    output_graph=agent.global_model.a,
    action_space_size=(agent.action_space,),
    tf_sess=agent.sess,
    # 开启红绿灯
    ignore_traffic_light=False)
# 然后agent先启动
agent.start()

# 之后以一定频率每次给一个数据集训练,state随机,但是action只是油门1,转向0,预期学习到在任何情况下都是油门1,转向0
while 1:
    # carla train里面内置训练和频率控制函数
    carla_model_trainer.step()
