# ReinforcementLearningInAutoPilot
ReinforcementLearningInAutoPilot
![](./Agent.png)
- 暂未开源
## 流程记录
- 安装carla仿真环境
- 封装carla的API以方便调用
- 根据RL任务写State，Action，Reward等的算法，注意这些模块输入不依赖于carla API
- 用上面的这些S A R组成Env，Env和gym尽可能相似的接口，env调用carla接口
- 重写开源代码的model和Agent流程，开始训练

## RL任务
- 进行RL是递进的过程，从简单任务逐渐到复杂任务
### 路径跟随
1. 给一个距离25米的直线，以车辆到直线偏离距离，到终点距离，速度等计算reward，给出state
2. 在1的基础上，进行连续RL，设置好间隔5m的waypoints，每次让车走直线到达下一个waypoint（连续RL体现在到达下一个waypoint后不会结束，而是更新target，继续从当前状态开始，直到到达最终的waypoint才reset，过程中出现了异常就回到最初，因为采用和相邻两个waypoint的垂直距离作为state和reward等的计算依据，因此更改waypoint不会对S和R的计算产生很大变动）
3. 在2的基础上，加上到达每个waypoint的速度限制，这个速度需要作为reward和state
