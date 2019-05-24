# ReinforcementLearningInAutoPilot

## 效果展示

![多路径点A3Cv2](./Results/pathFollowA3C_v2.gif)
<center>多路径点A3Cv2</center>

![单路径点DQNv1](./Results/RL_Curve.gif)

<center>单路径点DQNv1</center>

## 设计框架
![](./Agent.png)
- 暂未开源
## 运行流程
### 使用docker(推荐)
- 如果没有可视化需求,可以使用docker进行训练,开多个线程,每个线程设置不同的ip和port链接不同的carla仿真环境进行训练,提高训练效率
- carla-docker的安装可以参考carla官方文档,先安装docker,然后安装nividia docker,然后启动,设置map等
- docker的启动脚本见dockerUtils.py，可以同时启动多个server对应不同端口，然后在agent中填上对应主机的ip和对应端口（但是实测每台机器最好只开一个端口，多了存在通信问题）
### 使用可视化的UE
- 安装Carla的PythonAPI和CarlaUE环境(可根据carla官方文档),推荐找已经编译好的carla环境复制(查看"同平台的carla迁移")
- 建议使用Pycharm,建立新的Project并选定此文件夹作为工作目录（若不使用Pycharm，则将此文件夹加入环境变量）
- 启动Carla UE,打开一个map,然后运行或者运行模拟
- 在ReinforcementLearning/train文件夹下运行任意一个agent来进行测试：需要在相应的参数中填入编译好的PythonAPI路径,以及UE端的ip端口,然后运行
### 可视化UE同平台的carla迁移安装(以Ubuntu16.04相互迁移为例)
- PythonAPI只需要复制carla里面PythonAPI文件夹，然后尝试导入，解决依赖(遇到libpng16依赖可以再复制一下libpng16.so等文件)
- carlaUE需要复制carla整个文件夹，同时复制对应UE引擎，UE运行./Setup.sh，之后双击carlaUE里面的.uproject文件


## 建议的RL开发流程
1. 根据任务新建Environment，引用现有或者新建action，state等完成env搭建
2. 在搭建env过后，根据需求使用Model中的算法或者重新在Model中建立一个模型，写一个agent进行训练
3. 如果2中搭建出的agent训练效果较差,强烈推荐先大量测试开源算法,或者复制已有的agent改env进行测试,在效果良好的情况下再继续优化,如果大量算法测出来效果不好,进行env的debug
### 开源算法的迁移
- github上的RL算法代码大多是采用gym作为Env,因此,大部分的代码copy下来只需要修改env的部分即可,模型可以模块化放在Models中
### 建议的RL debug操作
- 首先确保env的各个数值和仿真引擎的对应没有问题，之后收集训练过程中的数据，分析模型和env中数值内在关系的问题，具体而言:
- 1.在初期调试的时候使用DataMonitor观察state，reward等是否和仿真引擎观测到的状态相符，观察的state，action数值范围，变化趋势，分布等是否合理(比如尽可能值域在-1到1，随时间线性的改变，连续).或者收集数据离线观察
- 2.作曲线图观察state-reward，action-reward等的相关性，确保符合设计预期
- 3.作state-action曲线图检查模型是否做出了符合预期的操作
- 4.单变量分析:对训练较长时间的模型，固定其他state，只变动一个state，构造state值域内的虚假数据集，预测action，绘制state-action曲线观察模型对于边界state的处理
- 5.观察分布:由贝叶斯公式P(action|state) = (P(action)P(state|action))/P(state),如果action设置得当,先验概率P(action)是等概率的,P(state|action)和P(state)可以用DataAnalysis中的工具对收集到的数据作出分布,由此可以验证不同state状态下是否做出了正确的action


## 注意事项
### 可视化UE运行FPS
- carlaUE的libcarla的代码执行频率应当是和UE的FPS相同的,在UE中打开FPS显示,尽可能让FPS高一些,可以通过项目设置中的carla设置进行调整,UE聚焦时windows下FPS能够达到120,非聚焦下70~80,尽可能将摄像头移动到消耗FPS少的地方,比如放大摄像头,移动到草地等
- UE的运行状态非常影响训练效果,因此最好在稳定的运行环境下训练,UE运行时不要最小化它,会因为UE的节省资源行为而出错,最好的方式是将窗口缩小放在能看到的地方
### 多车训练
- 多车同时训练采用多进程或多线程,每个车辆只需要创建一个env,然后采样得到action,state,reward等,然后push给全局模型去训练,之后pull回更新的权重
- 关于python的多进程和多线程参考:http://python.jobbole.com/86822/,因此最好选择多进程而非多线程
- 建议压力测试最大可同时训练的agent数目,并且评估通信延迟,action计算延迟,action到state的延迟等
- 目前的测试发现,模型训练,权重更新以及从UE中通信获取状态的延迟不高,但是多车过后因为通信阻塞的问题,UE端会出现部分车辆没有被控制的现象,可以尝试开几个docker,每个UE端启动几辆车进行训练
- 其他内容见：环境-并行


## 模块和RL的概念
模块化的目的
1. 去除引擎接口依赖:通过引擎接口将引擎和RL整个项目分离,实现替换引擎后(比如carla替换成AirSim等),所有RL项目内容无需改动,只需要改动对应的引擎接口engineInterface
2. 减少Env代码冗余:通常一个RL任务对应于一个Env,RL任务存在相似性,因此Env会存在大量冗余代码,所以使用模块化的Action,State,Reward等来构建Env,同时这些模块化的组件不依赖于engine,只有Env依赖于engine
3. Env兼容Agent:Env需要尽可能按照gym的接口进行实现,保证即使是github上面copy下来的开源算法只要更改了Env就能够立即使用!
4. Agent兼容Env:所有Agent都留出一个Env的输入参数,用于对任何Env都能够使用
5. Model的模块化:Model只依赖于来源机器学习/深度学习等算法框架,作为公用库引用,但是由于算法的特殊性,一些Model是完全只会在一个Agent中被引用的
### ./engineInterface
- 引擎接口提供车辆生成,接受控制参数,提供车辆信息和地图信息等,EngineInterface中规定所有引擎,如airsim,carla等需要实现的接口,接口文档存放在其中
### ./ReinforcementLearning/train
- 存放RL的训练代码,可以作为项目的入口
### Agent@./ReinforcementLearning/Modules/Agents
- Agent是算法承载的主体,依赖于Model,agent需要和输入的环境Env交互,得到状态State,将状态传入模型Model中,得到模型给出的动作Action,将Action放入Env中得到State,完成一轮循环
### 环境Env@./ReinforcementLearning/Modules/Environments
#### 并行
- 并行进行强化学习的训练对于效率提升很重要，需要环境能够支持并行
- 强化学习的环境需要基于仿真引擎来建立，因此仿真引擎需要支持并行的环境，比如能够开启很多server来提供给算法client进行训练。
- carla支持在地图中创建多个车辆进行学习，同时也能够使用docker在本机创建server通过不同的端口连接，即使一个端口也能在保证5个算法client进行连接，在局域网内部可以跨不同的机器进行连接。
- 并行需要评估训练过程的延迟，以保证每次的RL决策频率的稳定，比如在一个没有并行client在action过后，到下一个循环的用时3ms，如果RL是10hz，那么还需要sleep 97ms。而一个并行的client可能每次间隔20ms，此时需要动态修改等待的时间
- 另一个问题是server端的阻塞问题，如果python通过ipc调用carla的频率过高，则会出现阻塞导致延迟等情况，实测会出现车辆不被api控制的情况，可以通过减少RL运行频率，或者减少一个端口，或者一台机器server中所连接车辆的数目解决 
#### 接口和模块化 
- 为和开源代码兼容,所有Environments都实现IEnv中的接口,IEnv和gym.env接口相同,兼容大多数以gym为env的开源算法
- 考虑到迭代可能产生各种RL任务,分别需要的Env种类较多,为尽可能避免代码重复,Env都采用引用Action,Done,Rewards和States等模块的方式组成
- Action,Done,Rewards和States等都是不依赖于Engine的纯计算模块
- Env可能有很多任务,比如路径跟随,过红绿灯,多车避障,图像端到端等等,模块化的方式有利于单个任务的测试和任务的集成
### 模型Models@./ReinforcementLearning/Modules/Models
- 纯机器学习或神经网络模型,输入只有Action和State向量的长度,用于搭建模型,调用时给出Reward,Action,State的batch用于训练
- 需要实现存储,载入等方法
### 数据分析工具@@./ReinforcementLearning/Modules/DataAnalysisTools
- 用于在线的state，action，reward等数值监测，以及离线数据收集和分析
### ProcessManager@./ReinforcementLearning/Modules/ProcessManager
- 每个agent都需要用到的控制模块,给出随机进行action选择还是使用模型进行选择,是否应该进行训练,是否应该进行模型存储
- 退火e-greedy算法:以一定概率e进行随机操作,这个概率随着训练步数增加而减小,一开始随机性高,偏广度,之后随机性少,模型得到了充分训练,更为深度
### Archive@./ReinforcementLearning/Archive
- 一个RL任务训练表现较好后,在Archive中创建文件夹,将agent的代码复制进去,并带上相应的权重文件,要求运行agent时是演示而不进行训练





## 算法-模仿学习
### 模仿学习数据集和预训练模型
- 模仿学习是采用人类行为或者更成熟的算法根据state以及给出的action作为数据集,训练模型从state中得到action.action可以是直接的油门,转向甚至PID信息,也可以是处理过后的离散数据如左转,加速,减速等
- 数据来源可以是rosbag或者仿真器
- 对于rosbag或者仿真器,提取出需要的state和action,需要和仿真环境env中可以得到的state以及可以应用的action对应!

### GAIL算法概述和流程
- IL是模仿学习(Imitation Learning),GAIL相当于GAN+IL,模仿的对象的数据集是state+action作为X,得到判别器去判断state+action的组合是不是一个高级控制器(或者人类)控制的结果Y(二分类),然后使用一个生成器去基于state来生成action,得到的state-action对需要通过判别器鉴定,判别器尽可能优化使得能够区分出哪些是人类的操作,哪些是生成器给出的操作,生成器则尽可能得到能够让判别器认为是人类操作的state-action
#### GAIL开发流程
1. 获得模仿学习的数据集,state+action数据集{s+a}
2. 以s+a数据集作为X,1作为y构建真实数据{X: s+a , y: 1}
3. 以仿真器中采样的state,或者直接使用模仿学习的state作为生成器的输入,通过生成器得到action,将生成器得到的state+action作为X,以0为y构建虚假数据{X: s -> model -> a,y: 0}
4. 训练生成器使得能够使判别器判别为1,训练判别器使得能够使虚假数据判别为0,这个过程中生成器需要不断生成数据
5. 训练结束以后,用生成器作为模型,在模拟环境env中,给出state,得到模型的action进行测试{X:s, y:a}
6. 上述过程是离线训练的,在线训练可以持续给出真实数据集(人类行为纠正),持续进行GAIL训练,持续进行模型训练,或者交替模型/人类控制.

## 算法-强化学习
### DQN算法概述
- 神经网络输入state，输出和action长度相同的向量，输出向量是分别进行相应的action所获得的收益Q，每次可以贪心取最大的对应的action作为模型给出的建议action
- cost计算：上一个模型输入是readout，是网络根据state得到的每个action的Q值，将这个Q的向量和实际的action相乘（action是one-hot编码），最终得到输出，这个输出将和reward作方差，作为训练误差
- 使用replay memory来存储(state(t),action,state(t+1),reward,done)，一般一次滑动窗口存储多个（实际存储不会存储st+1，而是在采样时加进去）
- 随机从replay memory中拿出样本batch训练神经网络
### A3C算法概述
- DQN是Value-based,而A3C是Value和Policy Based都有,两个模型权重共享(多任务学习)
- Value Based用于预测Reward,Policy Based用于预测各个Action的概率,最终决策是使用概率
- A3C的网络模型框架很简单,核心在于loss的计算算法
### 遗传算法概述和流程
- 每辆车直接用NN模型回归到最后的准确操作（如油门，转向，刹车大小，挡位等,注意操作互斥）,而不是到DQN,A3C的离散操作(如转向-.25，油门+.25这种)
- 权重的优化使用遗传算法，reward越高，后代这种权重的比例越大，权重会浮点数随机变异
- 参考一个unity引擎中坦克大战的强化学习，每辆车state只有最近敌人的距离角度状态，用NN映射成前进和转弯，以及开炮等，使用遗传算法调整权重,因为一次有很多辆坦克,所以样本量大
- 缺点是优化慢,但是可以在模仿学习得到的预训练模型的基础上优化!
#### 流程
1. 获得模仿学习的数据集,训练一个从state直接到油门转向的预训练的模型
2. 使用遗传算法优化模型中的参数




## 强化学习任务搭建
- 一个完全的自动驾驶RL模型可能是从图像输入和其他传感器到车辆的最终控制信号,但这样的模型训练需要大量的样本,并且可能很难落地.低风险的方法是将RL应用于一些小场景,在每个场景中达成任务,然后逐步融合State,Action等形成大的任务
- 比如实现输入waypoints给出准确的油门转向控制,输入障碍物信息给出绕障的油门转向,然后将这两个任务结合,搭建路径跟随+绕障的模型
- 一个RL任务会新建一个Environment,并增量更新Action,State等模块
### 单车路径跟随
1. 给一个距离25米的直线，以车辆到直线偏离距离，到终点距离，速度等计算reward，给出state
2. 在1的基础上，进行连续RL，设置好间隔5m的waypoints，每次让车走直线到达下一个waypoint（连续RL体现在到达下一个waypoint后不会结束，而是更新target，继续从当前状态开始，直到到达最终的waypoint才reset，过程中出现了异常就回到最初，因为采用和相邻两个waypoint的垂直距离作为state和reward等的计算依据，因此更改waypoint不会对S和R的计算产生很大变动）
3. 在2的基础上，加上到达每个waypoint的速度限制，这个速度需要作为reward和state
### 多车路径跟随和车联网
1. 在路径跟随的基础上，添加state向量，以另一辆车的当前方向角和距离，另一辆车的速度方向角和距离作为新增的state，不需要以车辆距离作为reward，只需要一碰撞就设置为done（否则可能会和车道线跟随效果的持续reward产生竞争）
2. 需要两个agent进行，初始生成在相邻的车道，同时路径点平行
### 单车避障
1. 可以仿照多车路径跟随的思路设计，也是碰到障碍物之后done
### 单车路径跟随+速度限制
- 在state中添加下一个waypoint速度限制的特征，并修改reward（之前reward是速度越大越好，现在是速度越接近规定值越好），注意，因为速度限制的改变会让reward突变，因此速度改变后应当设置成一个新的RL任务，参考连续RL的思路
