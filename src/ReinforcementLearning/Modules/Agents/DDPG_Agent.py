# coding:utf-8
import tensorflow as tf
import numpy as np
import time
from ReinforcementLearning.Modules.Environments.IEnv import IEnv
from ReinforcementLearning.Modules.Models.Models import DDPG_Model_v1, DDPG_Global_And_Local_Models_v1, \
    DDPG_Model_v2_With_Reward_PreCorr
from ReinforcementLearning.Modules.Agents.IAgent import IAgent
from ReinforcementLearning.Modules.DataAnalysisTools.DataMonitor import RuntimeLineChart, LineConfig
import threading
from multiprocessing import Process
import warnings
import random
import copy


class DDPG_Agent_v1(IAgent):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    简单的DDPG测试模型,没有并行和存储功能
    '''

    def __init__(self, env,
                 save_dir="./ddpg_ckpt/",
                 save_interval=1000,
                 use_model_with_pre_calculated_g=True,
                 kwargs_for_model=None
                 ):
        if kwargs_for_model is None:
            kwargs_for_model = {}
        self.s_dim = env.observation_space[0]
        self.a_dim = env.action_space[0]

        if use_model_with_pre_calculated_g:  # 使用预先计算的q值作为value
            self.model = DDPG_Model_v2_With_Reward_PreCorr(
                a_dim=self.a_dim, s_dim=self.s_dim, a_bound=1.0, **kwargs_for_model)
        else:
            self.model = DDPG_Model_v1(a_dim=self.a_dim, s_dim=self.s_dim, **kwargs_for_model)
        self.env = env
        self.env.connect()
        self.env.start()

    def train(self):

        # 在action输出时,对输出的浮点数进行一次随机,这个是随机的方差
        variance = 1
        # 随机的方差会逐渐减小,以从广度过渡到深度/确定性搜索
        variance_decay = 0.995
        # 当完成这个步数的时候variance *= decay
        variance_decay_step = 100
        current_state = self.env.reset()
        total_step = 1
        # ep_reward = 0
        while 1:
            action = self.model.choose_action(current_state)
            # action分别是油门/刹车 和 左转/右转  因此值域-1到1,这里加上方差
            action = np.clip(np.random.normal(action, variance), -1, 1)
            new_state, reward, done, _ = self.env.step(action)

            self.model.store_transition(current_state, action, reward / 10, new_state)
            if total_step % variance_decay_step == 0:
                variance *= variance_decay

            current_state = new_state
            # ep_reward += reward
            total_step += 1
            if done:
                self.model.learn()
                print('Episode:', total_step, ' Reward: %i' % int(reward), 'Explore: %.2f' % variance,)


class DDPG_Agent_GAL_v1(IAgent):
    Author = "BaoChuan Wang"
    AllowImport = True

    def __init__(self, env_prototype_dict_for_workers,
                 save_dir="./ddpg_ckpt/",
                 save_interval=100, model_hook_dict=None,
                 # 下面的kwargs每一个用env_prototype_dict_for_workers的name作为key,传给worker或者model的kwargs字典作为value
                 kwargs_for_worker_dict=None,
                 kwargs_for_model_dict=None,
                 kwargs_for_global_model=None
                 ):

        assert isinstance(env_prototype_dict_for_workers, dict)
        first_key = list(env_prototype_dict_for_workers.keys())[0]
        assert isinstance(env_prototype_dict_for_workers[first_key], IEnv)
        self.action_space = env_prototype_dict_for_workers[first_key].action_space[0]
        self.observation_space = env_prototype_dict_for_workers[first_key].observation_space[0]
        self.save_dir = save_dir
        self.save_interval = save_interval
        if kwargs_for_worker_dict is None:
            kwargs_for_worker_dict = {}
            print("[WARNING]" + self.__class__.__name__ + ": No kwargs for workers!")
            time.sleep(1)
        if kwargs_for_model_dict is None:
            print("[WARNING]" + self.__class__.__name__ + ": No kwargs for models!")
            kwargs_for_model_dict = {}
            time.sleep(1)
        # 所有模型,统一sess
        self.sess = tf.Session()
        if kwargs_for_global_model is None:
            kwargs_for_global_model = {}
        # 全局模型加载
        self.global_model = DDPG_Global_And_Local_Models_v1(
            is_global_model=True,
            a_dim=self.action_space,
            s_dim=self.observation_space,
            scope="global",
            tf_sess=self.sess,
            save_dir=save_dir, **kwargs_for_global_model)
        self.global_model.load()

        self.workers = []

        for name in env_prototype_dict_for_workers:
            model_hook = None
            if model_hook_dict is not None:
                if name in model_hook_dict.keys():
                    model_hook = model_hook_dict[name]
            env = env_prototype_dict_for_workers[name]

            if name in kwargs_for_model_dict.keys():
                kwargs_for_model = kwargs_for_model_dict[name]
            else:
                kwargs_for_model = {}
            if name in kwargs_for_worker_dict.keys():
                kwargs_for_worker = kwargs_for_worker_dict[name]
            else:
                kwargs_for_worker = {}

            local_model = DDPG_Global_And_Local_Models_v1(
                is_global_model=False,
                global_model=self.global_model,
                a_dim=self.action_space,
                s_dim=self.observation_space,
                scope=name,
                tf_sess=self.sess,
                save_dir=save_dir,
                **kwargs_for_model
            )
            self.workers.append(DDPG_GAL_Worker_v1(
                env=env,
                name=name,
                global_model=self.global_model,
                local_model=local_model,
                save_ineterval=save_interval,
                tf_sess=self.sess, model_hook=model_hook, **kwargs_for_worker))

    def add_worker(self, env, name, kwargs_for_model, model_hook, kwargs_for_worker):
        local_model = DDPG_Global_And_Local_Models_v1(
            is_global_model=False,
            global_model=self.global_model,
            a_dim=self.action_space,
            s_dim=self.observation_space,
            scope=name,
            tf_sess=self.sess,
            save_dir=self.save_dir,
            **kwargs_for_model
        )
        self.workers.append(DDPG_GAL_Worker_v1(
            env=env,
            name=name,
            global_model=self.global_model,
            local_model=local_model,
            save_ineterval=self.save_interval,
            tf_sess=self.sess, model_hook=model_hook, **kwargs_for_worker

        ))

    def start(self):
        coord = tf.train.Coordinator()
        self.sess.run(tf.global_variables_initializer())
        worker_threads = []
        for worker in self.workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)  # 创建一个线程，并分配其工作
            t.start()  # 开启线程
            worker_threads.append(t)
        # 这里不要等待线程join,因为外面还有主线程!
        # coord.join(worker_threads)  # 把开启的线程加入主线程，等待threads结束


class DDPG_GAL_Worker_v1(object):
    Author = "BaoChuan Wang"
    AllowImport = True

    # 下面的参数用于计算平均reward
    # 总reward
    TOTAL_REWARD = 0
    # 从开始到done的步数
    TOTAL_STEP = 0
    # 总worker数目
    TOTAL_WORKER_NUM = 0

    def __init__(self, env, name,
                 # 本地和全局模型,tf计算图
                 local_model,
                 global_model,
                 tf_sess,
                 # 是否进行RL学习,如果是纯粹模仿学习可以关闭
                 do_RL_learn=True,
                 # 存储间隔,是done多少次之后save
                 save_ineterval=100,
                 # ddpg是采用数值的输出,因此需要添加一定均值和方差的高斯加到action输出上,以便于引导ddpg进行探索
                 # 初始对于每个action的方差,在action输出时,对输出的浮点数进行一次随机,这个是随机的方差
                 start_variance_for_each_action=(1.0, 1.0),
                 # 方差每次衰减的比例,随机的方差会逐渐减小,以从广度过渡到深度/确定性搜索
                 variance_decay_ratio_for_each_action=(0.995, 0.995),
                 # 方差每次经过多少步骤衰减
                 variance_decay_step=100,
                 # 初始对于每个action的修正值
                 start_offset_for_each_action=(0, 0),
                 # 每次修正值的减少量,直接和下面的值相加
                 offset_decay_value_for_each_action=(-0.01, -0.01),
                 # 每间隔多少步对这个修正量进行减少,减少到0就是action完全对应的输入
                 offset_decay_step=100,
                 # 用以注入其他agent/model参数的hook
                 model_hook=None,
                 debug=False):
        self.worker_index = self.__class__.TOTAL_WORKER_NUM
        self.__class__.TOTAL_WORKER_NUM += 1
        self.do_RL_learn = do_RL_learn
        if do_RL_learn == False:
            print("Worker %s will not do RL"% name)
        self.start_variance_for_each_action = np.array(start_variance_for_each_action)
        self.variance_decay_ratio_for_each_action = np.array(variance_decay_ratio_for_each_action)
        self.variance_decay_step = variance_decay_step
        self.start_offset_for_each_action = np.array(start_offset_for_each_action)
        self.offset_decay_value_for_each_action = np.array(offset_decay_value_for_each_action)
        self.offset_decay_step = offset_decay_step
        # 保证方差和修正的数量和action数目匹配
        assert env.action_space[0] == self.start_variance_for_each_action.shape[0] == \
               self.variance_decay_ratio_for_each_action.shape[0] == \
               self.start_offset_for_each_action.shape[0] == \
               self.offset_decay_value_for_each_action.shape[
                   0], "Should set var, offset and their decay for each action!"

        self.offset_for_each_action = np.array(self.start_offset_for_each_action)
        self.variance_for_each_action = np.array(self.start_variance_for_each_action)
        self.env = env
        self.env.connect()
        self.name = name
        self.local_model = local_model
        self.sess = tf_sess
        self.global_model = global_model
        self.save_interval = save_ineterval
        self.model_hook = model_hook
        self.debug = debug
        self.env.start()
        if debug:
            # 用于实时变量监控的monitor,创建线型参数
            self.monitor_line_config = {
                "throttle or brake": LineConfig(color=(1, 0, 0), line_marker=LineConfig.LineMarkerStar),
                "turn": LineConfig(color=(0, 1, 0), line_marker=LineConfig.LineMarkerStar),
                "reward": LineConfig(color=(0, 0, 1), line_style=LineConfig.LineStyleDashDot), }

            self.monitor_data = {}
            # monitor的数据，初始化
            for vname in self.monitor_line_config:
                self.monitor_data[vname] = 0
            self.monitor = RuntimeLineChart(ylim=(-1, 1), window_area=100,
                                            vname_to_line_config_dict=self.monitor_line_config)
            # 需要在另一个进程中开启
            p = Process(target=self.monitor.run)
            p.start()

    def work(self):
        # local从global中拉取权重!
        self.local_model.pull_global()
        current_state = self.env.reset()
        total_step = 1
        # ep_reward = 0
        while 1:
            # 如果有hook,就会截取state来替换成hook给出的action
            if self.model_hook is not None:
                action = self.model_hook.tamper_action(self.env, current_state)
            else:
                action = self.local_model.choose_action(current_state)
                model_action = copy.deepcopy(action)
                for i in range(action.shape[0]):
                    # action分别是油门/刹车 和 左转/右转,因此值域-1到1,这里先随机,然后加上offset
                    action[i] = np.clip(
                        np.random.normal(
                            action[i],
                            self.variance_for_each_action[i]) + self.offset_for_each_action[i],
                            -1, 1)
                if random.randint(0, 50) == 0:  # 以1/50概率打印模型输出
                    print("Model predict action:", model_action, "After randomed", action)

            new_state, reward, done, _ = self.env.step(action)
            if self.debug:
                # 更新monitor数据
                self.monitor_data["throttle or brake"] = action[0]
                self.monitor_data["turn"] = action[1]
                self.monitor_data["reward"] = reward
                self.monitor.update_data(self.monitor_data)
                # print("Action: %s Reward %s"%(action,reward))

            self.local_model.store_transition(current_state, action, reward, new_state)
            # print(":",total_step % variance_decay_step)
            # 方差和offset衰减
            if total_step % self.variance_decay_step == 0:
                self.variance_for_each_action = self.variance_for_each_action * self.variance_decay_ratio_for_each_action
            if total_step % self.offset_decay_step == 0:
                self.offset_for_each_action = self.offset_for_each_action - self.offset_decay_value_for_each_action
                for i in range(self.offset_for_each_action.shape[0]):
                    if self.offset_for_each_action[i] < 0.0:
                        self.offset_for_each_action[i] = 0.0

            # print(self.variance_for_each_action,self.offset_for_each_action)
            current_state = new_state
            # ep_reward += reward
            total_step += 1
            self.__class__.TOTAL_REWARD += reward
            # if done or total_step % 30 == 0:
            if done:
                self.__class__.TOTAL_STEP += 1
                print("Now global step", self.__class__.TOTAL_STEP)
                if self.do_RL_learn:
                    print("Give global to learn!")
                    self.local_model.give_global_to_learn()
                if self.__class__.TOTAL_STEP % self.save_interval == 0:
                    self.global_model.save(global_step=self.__class__.TOTAL_STEP)
                    print("Worker of index %s saved model weights" % self.worker_index)
                # 更新后拉取权重
                self.local_model.pull_global()
                print("worker%s: " % self.name, 'Step: ', total_step, ', Now Reward: %.5f' % reward,
                      "Global Mean Reward %.2f" % (self.__class__.TOTAL_REWARD / self.__class__.TOTAL_STEP),
                      'Explore mean variance %.5f, mean offset %.5f' % (
                          float(np.mean(self.variance_for_each_action)),
                          float(np.mean(self.offset_for_each_action))))
                # print("Dataset num%s" % self.local_model.pointer)
                # 建议不要clear掉历史
                # self.local_model.clear_memory()
