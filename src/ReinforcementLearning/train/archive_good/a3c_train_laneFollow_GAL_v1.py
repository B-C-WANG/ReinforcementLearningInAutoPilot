# coding:utf-8
# Type: Private Author: BaoChuan Wang

'''
描述:
使用A3C算法,沿着车道行驶,直行和过弯基本上都能够在车道内,相比waypoints效果好
但是转弯有漂移,并且不减速,速度快了效果差
只要加上速度控制就会较好!
学习不到减速是RL的最大问题,很多文章都遇到了这个问题

心得: 车道线限制过大学习效果下降,可以不对算法车道线横向距离限制,将0.15增加
'''

from ReinforcementLearning.Modules.Agents.A3C_Agent import A3C_GAL_Train_Agent_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import HighPrecisionControl_v1

# 给每个worker创建一个env

# docker的启动见dockerUtils.py
# server端口设置，key是ip，value是每个端口,可以使用docker启动多个server，每个在不同端口,或者链接不同的主机，填入ip和主机中的port
# 推荐配置:一个主要的可视化的server+几个docker的server

# key是ip,value是提供的端口,每个端口对应一个UE环境
server_config = {
    # 建议每台机器只开一个server
    # "10.10.9.214": [2000],
    "10.10.9.128": [2000],
}

# 每个端口多少的worker,也是一个UE环境同时进行训练的车辆数目,数量过多在UE会有很大延迟,需要注意
n_workers_in_each_port = 5

# 在每个server中每辆车初始生成的位置list,数量需要大于n worker in each port
spawn_index_for_each_car_in_worker = (0, 10, 20, 30, 40, 50, 60,70,80)

env_dict = {}

for ip in server_config:
    for port in server_config[ip]:
        for i in range(n_workers_in_each_port):
            name = 'W_%s' % (str(ip) + "_" + str(port) + "_" + str(i))  # worker name
            # 拿到最后一个worker的name,用于延迟debug
            delay_debug_worker_name = name

            env = LaneFollowEnv_v1(
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip=ip,
                carla_UE_port=port,
                vehicle_start_point_index=spawn_index_for_each_car_in_worker[i],
                wait_time_after_apply_action=0.1,
                # 使用高精度控制的action
                action_replace=HighPrecisionControl_v1(),
                # 对于a3c,车道线的限制不需要太严格,否则也很难训练!
                maximum_distance_to_lane=10,
            )
            env_dict[name] = env

delay_debug_worker_name = []

# 用于延迟debug的worker name,注释掉以关闭最后一个worker进行delay debug
#delay_debug_worker_name = [delay_debug_worker_name]

# 最后用name:env的字典传入env,env用于agent训练,每个env分配一个worker
A3C_GAL_Train_Agent_v1(env_prototype_dict_for_workers=env_dict, save_dir="./a3c_gal_ckpt/",
                       delay_debug_worker_name=delay_debug_worker_name).start()
