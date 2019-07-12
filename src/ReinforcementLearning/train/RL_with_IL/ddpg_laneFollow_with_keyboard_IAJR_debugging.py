# coding:utf-8
# Type: Private Author: BaoChuan Wang

'''

FIXME:失败:键盘提供的高reward行为没有被RL学习到

ddpg进行强化学习,同时从人类的键盘控制的行为中进行模仿学习
将人类键盘控制的这部分的learning rate加大,或者梯度加大,用于明显观察到
模仿学习的效果!

实际上键盘控制的不一定有DDPG自己算法的好,
所以主要是需要观察能不能够学习到键盘行为!

预期结果:键盘控制的车辆如果一直直行能够得到好的reward,那么RL控制的车辆也会试图直行

测试1:
一开始键盘车不动,RL车不动
键盘车前进多次然后done过后,RL的车仍然不动
并且预测的action反而是减速状态增多,
已经确认在有油门的情况下,q值为-0.2,loss为0.2很小,而没有油门loss是0.5
但是NN没有优化到

测试2:
将RL和键盘车的importance都改成0.0,预期每次预测出的action相同,实际上action会有改变

测试3:
打印出数据集,结果发现数据集正常, 采用0.5油门作为action时的q值最大,为-0.05,其他情况下没有油门为-0.5
逐渐优化发现,最后预测的action,油门和转向都逼近1

测试4:
只把模仿的键盘的agent对应的worker的RL学习功能(do_RL_learn)打开,关闭其他所有agent的RL学习



'''

import random
import numpy as np
from ReinforcementLearning.Modules.Agents.DDPG_Agent import DDPG_Agent_GAL_v1
from ReinforcementLearning.Modules.Environments.Environments_laneFollow import LaneFollowEnv_v1
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1
from ReinforcementLearning.ImitationLearning.ModelHooks.ModelHooks import KeyBoardHook_to_throttle_or_brake_and_steering
from ReinforcementLearning.Modules.Environments.Rewards import SafeDriveDistanceCost

# 关于以下设置,请参考最初的A3c train waypoints global and local
server_config = {

    "10.10.9.128": [2000],
}
# 由于没有控制IL的模型的lr,所以可能RL模仿IL的效率低,因此车辆总数少一些,因为IL只有一个
n_workers_in_each_port = 2

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
                use_random_start_point=True,
                wait_time_after_apply_action=0.1,
                # 这里是DDPG和A3C算法的区别,使用连续空间的action
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
                "gamma": gamma,
                "use_pre_calculated_g": use_pre_calculated_g,
            }
            worker_kwargs[name] = {
                # 因为是模仿为主,所以RL不进行探索
                "start_variance_for_each_action": (0., 0.),
                "variance_decay_ratio_for_each_action": (0.995, 0.995),
                "variance_decay_step": 10,
                "start_offset_for_each_action": (0.0, 0.0),
                "offset_decay_value_for_each_action": (0.00, 0.00),
                "offset_decay_step": 2000,
                # 将RL学习关闭(之后会只打开模仿对象的RL学习)
                "do_RL_learn":False
            }

hook_dict = {}


# keyboard hook控制
kh = KeyBoardHook_to_throttle_or_brake_and_steering()

# 随便选择一个agent加上keyboard hook,因为传入的是字典
keyboard_agent_name = random.choice(env_dict.keys())
hook_dict[keyboard_agent_name] = kh

# 把键盘控制的车辆放到指定的点,同时关闭随机起点
env_dict[keyboard_agent_name].vehicle_start_point_index = 0
env_dict[keyboard_agent_name].use_random_start_point = False

# 把模仿对象的RL功能打开
worker_kwargs[keyboard_agent_name]["do_RL_learn"] = True


# 键盘控制的对象进行debug,观察是否模型进行了训练!
# worker_kwargs[keyboard_agent_name]["debug"] = True

DDPG_Agent_GAL_v1(env_prototype_dict_for_workers=env_dict,
                  save_dir="./ddpg_IAJR_ckpt/",
                  model_hook_dict=hook_dict,
                  kwargs_for_model_dict=model_kwargs,
                  kwargs_for_worker_dict=worker_kwargs,
                  kwargs_for_global_model={
                      # 预先求得准确q值给数据集
                      "use_pre_calculated_g": use_pre_calculated_g,
                      "gamma": gamma}).start()
# 注意:如果下面的代码没有运行,是上面start过后有join函数等待

# 然后打开keyboard hook控制
kh.start_keyboard_control()

# 接下来就是找到键盘控制的车辆,持续用键盘修改油门刹车转向!

# 主线程等待join
while 1:
    pass
    # if random.randint(0,10) == 0:
    #   print(kh.throttle_or_brake)

# 然后观察现象: 把键盘控制的车辆一直加速度,观察其他RL控制的车辆是否也是一直加速
# 注意选择的方差减少一些
