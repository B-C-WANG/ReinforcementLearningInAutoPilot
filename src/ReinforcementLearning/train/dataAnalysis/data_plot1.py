# coding:utf-8
# Type: Private Author: BaoChuan Wang

import time
import numpy as np
import matplotlib.pyplot as plt
from ReinforcementLearning.Modules.DataAnalysisTools.DataAnalysis import DataAnalysis
# 载入数据
action = np.load("./data/action_35000to40000.npy")
reward = np.load("./data/reward_35000to40000.npy")
state = np.load("./data/state_35000to40000.npy")

print(action.shape)
print(reward.shape)
print(state.shape)

# 根据env的state提取state分量
target_xy_array_relate_to_car_x=state[:,0]
target_xy_array_relate_to_car_y= state[:,1]
car_forward_angle_diff =    state[:,2]
car_to_way_angle_diff =     state[:,3]
car_velocity_angle_diff =   state[:,4]
car_accel_angle_diff =      state[:,5]
accel =                     state[:,6]
velocity =                  state[:,7]



# 对各个变量随时间作图,观察值域和连续性

time_ = list(range(action.shape[0]))
plt.plot(time_,car_to_way_angle_diff)


# 观察速度方向和车辆方向的相关性

#plt.plot(car_velocity_angle_diff,car_forward_angle_diff,"ro")


plt.show()

# 观察不同action下的state的分布!
#plt.close()
DataAnalysis.plot_hist_on_different_ylabel(car_to_way_angle_diff,action).show()



