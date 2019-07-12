# coding:utf-8
# Type: Private Author: BaoChuan Wang
import os
import numpy as np
from ReinforcementLearning.Modules.Models.DecisionTreeModels import RegressionTreeModels
'''
尝试用决策树等机器学习方法对carla agent从state给出的action进行分析
'''

dirs = "../archive_good/laneFollowTrafficLightData/"
files = os.listdir(dirs)
all_state_data = {}
all_action_data = {}
for i in files:
    if i.endswith(".npy"):
        if i.startswith("action"):
            index = i.split("action_")[1].split("to")[0]
            all_action_data[index] = i
        elif i.startswith("state"):
            index = i.split("state_")[1].split("to")[0]
            all_state_data[index] = i
print(all_action_data.keys())
print(all_state_data.keys())

action_data = []
state_data = []
for i in all_action_data.keys():
    action_data.append( np.load(dirs+all_action_data[i]))
    state_data.append(np.load(dirs+ all_state_data[i]))

action_data = np.concatenate(action_data,axis=0)
state_data = np.concatenate(state_data,axis=0)

# 决策树,训练很慢
# model = RegressionTreeModels()
# model.build()
# model.train(state_data[:1000],action_data[:1000])
# model.visualize_tree()

# gbr
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


model = GradientBoostingRegressor(n_estimators=10000)
X_train, X_test, y_train, y_test = train_test_split(state_data,action_data,test_size=0.33)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# GBR只能预测一个target,所以设置target的index
target_index = 0
model.fit(X_train,y_train[:,target_index])
test_pre = model.predict(X_test)
plt.plot(y_test[:,target_index],test_pre,"ro")
plt.show()
# 打印出各个feature的importance
print(model.feature_importances_)
'''
importance分析结果:
对于油门和刹车这个target,importance为
速度标量       0.45935925  
速度x         0.01006275  
速度y         0.00771682  
加速度标量     0.3719029   
加速度x        0.05408772  
加速度y        0.00396671
车道线c0       0.01144348  
车道线c1       0.00659045  
车道线c2       0.00827322  
车道线c3       0.02414223  
是否有红绿灯   0.00360852  
到红绿灯距离   0.03884595
由于拟合效果不好(尤其是对于在红绿灯下),因此importance不正确

对于转向,importance为
 速度标量       0.02484314  
 速度x         0.03301186  
 速度y         0.07547246  
 加速度标量     0.02352136  
 加速度x        0.02346913  
 加速度y        0.2898656
 车道线c0       0.0431133   
 车道线c1       0.14221561  
 车道线c2       0.30776324  
 车道线c3       0.02888994  
 是否有红绿灯    0.00094648  
 到红绿灯距离    0.00688788
影响较大的为横向加速度和车道线曲率,然后是车道线斜率

'''



