# coding:utf-8
# Type: Private Author: BaoChuan Wang

'''
描述:
使用A3C算法,同时并行5辆车进行训练,过路径点,效果一般
在训练约8h时,能够控制好在直线上,转弯不会过于发散
过弯时速度低能够很顺利,速度高会有一些问题

目前最大的问题就是车辆一直加速,不减速,低速下能够对方向有较好控制,高速度下很难,虽然能够观测到转弯减速,但不明显

'''

from ReinforcementLearning.Modules.Agents.A3C_Agent import A3C_GAL_Train_Agent_v1
from ReinforcementLearning.Modules.Environments.Environments_waypointTarget import CarlaConsecutiveWaypointsTargetEnv_v1
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

            env = CarlaConsecutiveWaypointsTargetEnv_v1(
                carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                carla_UE_ip=ip,
                carla_UE_port=port,
                n_waypoint=50,
                # 因为后面的设置加上了车道线信息,所以waypoints不需要设置太短,设置太短容易蛇形走位,
                # Waypoints主要目的是用来设置给出reward的checkpoint,waypoints可以间隔非常长,但是reward会因为长时间没有为正而导致没有办法学习
                waypoint_spacing=10,
                vehicle_start_point_index=spawn_index_for_each_car_in_worker[i],
                wait_time_after_apply_action=0.1,
                # 放宽到达waypoint的条件
                ratio_of_reaching=0.2,
                # 是否添加车道线的拟合参数作为原来state的扩增?
                add_center_lane_state=True,
                # 使用高精度控制的action
                action_replace=HighPrecisionControl_v1()
            )
            env_dict[name] = env

delay_debug_worker_name = []

# 用于延迟debug的worker name,注释掉以关闭最后一个worker进行delay debug
#delay_debug_worker_name = [delay_debug_worker_name]

# 最后用name:env的字典传入env,env用于agent训练,每个env分配一个worker
A3C_GAL_Train_Agent_v1(env_prototype_dict_for_workers=env_dict, save_dir="./a3c_gal_ckpt/",
                       delay_debug_worker_name=delay_debug_worker_name).start()
