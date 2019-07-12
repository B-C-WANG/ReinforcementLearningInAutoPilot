# coding:utf-8
# Type: MultiAuthor


from ReinforcementLearning.Modules.utils import History, ReplayMemory
from ReinforcementLearning.Modules.ProcessManager import PretrainLinearEGreedyAnnealingManager
from ReinforcementLearning.Modules.Models import DenseModel_v1
from ReinforcementLearning.Modules.Environments import CarlaConsecutiveWaypointsTargetEnv_v1
from ReinforcementLearning.Modules.utils import colored_reward
import numpy as np
import time


class DQNAgent_v1(object):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    每辆车都一个agent,但是每个agent可以都有一个同样的QEvalModel,或者同样的action等等共享(因为都是模块化的)
    注意如果使用多线程,多进程,保证对象一致!
    '''

    def __init__(self,
                 # 模型存储文件夹
                 ckpt_directory,
                 env,
                 # 输入输出并不是单个的state,而是前面几个frame的state组成的向量或者矩阵
                 n_record_frame=4):

        # 训练的环境,环境与UE通信
        self.env = env
        self.env.connect()
        # 每次训练的batch size,随机采样,所以用Memory池来存储,并且需要用预训练来增加池的数据量
        self.train_batch_size = 32
        self.memory = ReplayMemory(size=5000000, sample_shape=self.env.observation_space, history_length=n_record_frame)

        # state历史,用于提供frame个state
        self.history = History((n_record_frame,) + self.env.observation_space)

        # 用来决定使用随机预测还是模型预测,是否训练和是否存储模型
        self.process_manager = PretrainLinearEGreedyAnnealingManager(
            e_greedy_start=1,
            e_greedy_end=0.1,
            e_greedy_total_steps=100000,
            n_pretrain_steps=20000,
            n_train_interval_steps=4,
            n_model_save_interval=1000)

        # 权重存储的文件夹
        if not ckpt_directory.endswith("/"):
            raise ValueError("ckpt directory should end with /, like ckpt/")

        # 模型输入包括历史n_frame个的state,所以乘以state frame
        self.q_eval_model = DenseModel_v1(input_space_size=(self.env.observation_space[0] * n_record_frame,),
                                       output_space_size=self.env.action_space,
                                       hidden_size=(32, 16),
                                       gamma=0.9,ckpt_directory=ckpt_directory)

        # 衡量训练算法状态
        self.episode_rewards, self.episode_q_means, self.episode_q_stddev = [], [], []

    def start(self):
        self.env.start()
        self.q_eval_model.init_tf_weights()
        # 尝试读取保存的权重
        self.q_eval_model.load_tf_weights()


    def step(self, state):
        # history存储主要是为了凑成n_frame个state来输入
        self.history.append(state)

        if self.process_manager.should_use_random_action():
            action = np.random.choice(self.env.observation_space[0])

        else:
            # 一共n_frame个的state从history中拿出
            env_with_history = self.history.value
            # 输入是4倍的state,需要flatten成向量,然后预测q值
            q_values = self.q_eval_model.predict_q(env_with_history.reshape((1, -1)))
            # q_values = self.q_eval_model.predict_q(state.reshape((1,)+state.shape))

            # 模型效果
            self.episode_q_means.append(np.mean(q_values))
            self.episode_q_stddev.append(np.std(q_values))

            # 使用贪心选择q最大的
            action = np.argmax(q_values)
            print("NN Predict action: %s"%action)
        # action步数增加
        self.process_manager.action_step_plus()
        # 应用action并返回New state rewards等等
        return self.env.step(action)

    def observe(self, old_state, action, reward, done):
        # 将马尔科夫过程的参数加入memory用于训练
        self.episode_rewards.append(reward)
        if done:
            self.episode_rewards, self.episode_q_means, self.episode_q_stddev = [], [], []
            self.history.reset()
        self.memory.append(old_state, action, reward, done)


    def train_model(self):
        if self.process_manager.should_train():
            # 从memory中拿出一批数据训练
            pre_states, actions, post_states, rewards, terminals = self.memory.minibatch(self.train_batch_size)
            #  对于state的4个frame,这里是采用的flatten送入4倍的NN input中
            pre_states = pre_states.reshape(pre_states.shape[0], -1)
            post_states = pre_states.reshape(post_states.shape[0], -1)
            actions = actions.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)
            terminals = terminals.reshape(-1, 1)
            # make actions one hot
            action_one_hot = np.zeros((actions.shape[0], +(self.env.action_space[0])))
            action_one_hot[:, actions] = 1
            self.q_eval_model.train(
                pre_state_batch=pre_states,
                action_batch=action_one_hot,
                post_state_batch=post_states,
                reward_batch=rewards,
                terminal_batch=terminals
            )
            # 训练步数增加
            self.process_manager.train_step_plus()
            print("Trained", self.process_manager.steps_train_taken)
            # 保存模型
            if self.process_manager.should_save_model():
                self.q_eval_model.save_tf_weights()


    def train(self):
        self.start()
        # 给个时间看一下读取权重是否成功
        time.sleep(2)
        state = self.env.reset()
        while 1:
            new_state,reward,done,action = self.step(state)
            print(str(self.process_manager.steps_action_taken) + "\t" + colored_reward(reward))
            self.observe(state, action, reward, done)
            self.train_model()
            state = new_state
