# coding:utf-8
# Type: MultiAuthor

import threading
import tensorflow as tf
import numpy as np
import traceback

from ReinforcementLearning.Modules.Environments.IEnv import IEnv
from ReinforcementLearning.Modules.DataAnalysisTools.DelayAnalyser import DelayAnalyser
from ReinforcementLearning.Modules.Models.Models import A3C_Global_And_Local_Models_v1
from .IAgent import IAgent


class A3C_GAL_Train_Agent_v1(IAgent):
    Author = "BaoChuan Wang"
    AllowImport = True

    '''
    reference: 
    https://github.com/princewen/tensorflow_practice/blob/master/RL/Basic-A3C-Demo/A3C.py
    因为Environment接口非常统一,所以将上面的Agent改成自己所需要的Agent几乎没有费太多时间
    
    A3C global and locals
    这个agent会创建一个global模型和多个local模型,每个port会创建一个worker,加入一个env,然后训练
    训练done过后,梯度或者dataset传给global模型(push),然后global更新权重,local pull下来更新的权重!
    '''

    def __init__(self,
                 # dict,key为worker名称,value为env对象,每个worker一个env!
                 env_prototype_dict_for_workers,
                 # 模型权重存储路径
                 save_dir="./a3c_gal_ckpt/",
                 # 间隔多少步存储
                 save_interval=1000,
                 # 分析延迟,给一个worker的名称,该worker就会打印出主要循环步骤中的耗时
                 delay_debug_worker_name=None):

        assert isinstance(env_prototype_dict_for_workers, dict)

        # 拿出第一个env来得到输入输出大小,用于创建模型
        first_key = list(env_prototype_dict_for_workers.keys())[0]
        assert isinstance(env_prototype_dict_for_workers[first_key], IEnv)
        self.action_space = env_prototype_dict_for_workers[first_key].action_space
        self.observation_space = env_prototype_dict_for_workers[first_key].observation_space

        self.sess = tf.Session()

        # with tf.device("/cpu:0"):
        # 创建一个全局的模型,注意全局的scope
        self.global_model = A3C_Global_And_Local_Models_v1(is_global_model=True,
                                                           action_space=self.action_space,
                                                           observation_space=self.observation_space,
                                                           scope="global_model",
                                                           tf_sess=self.sess,
                                                           save_dir=save_dir)
        self.workers = []

        # 总网络来读取权重，所有权重读写都是总的网络进行！
        self.global_model.load()
        for name in env_prototype_dict_for_workers:
            delay_debug = False
            if delay_debug_worker_name is not None:
                if name in delay_debug_worker_name:
                    delay_debug = True
            env = env_prototype_dict_for_workers[name]
            try:
                # 从dict中获取name和对应的env,建立相应的Model,最终将Model和env传入worker,worker来训练
                local_model = A3C_Global_And_Local_Models_v1(action_space=env.action_space,
                                                             observation_space=env.observation_space,
                                                             scope=name,
                                                             global_model=self.global_model,
                                                             tf_sess=self.sess, save_dir=save_dir)
                self.workers.append(A3C_GAL_Worker_v1(name=name,
                                                      global_model=self.global_model,
                                                      env=env,
                                                      save_interval=save_interval,
                                                      local_model=local_model,
                                                      tf_sess=self.sess,
                                                      delay_debug=delay_debug))
            except:
                traceback.print_exc()

    def start(self):
        # 开始所有训练

        # Coordinator类用来管理在Session中的多个线程，
        # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
        COORD = tf.train.Coordinator()
        self.sess.run(tf.global_variables_initializer())

        # if OUTPUT_GRAPH:
        #     if os.path.exists("./log"):
        #         shutil.rmtree("./log")
        #     tf.summary.FileWriter("./log", self.sess.graph)

        worker_threads = []
        for worker in self.workers:
            # job是一个匿名的函数对象
            job = lambda: worker.work()
            # TODO:尝试多进程代替多线程?
            # Process(target=job).start()

            # 创建一个线程，并分配其工作
            t = threading.Thread(target=job)
            # 开启线程
            t.start()
            worker_threads.append(t)
        # 把开启的线程加入主线程，等待threads结束(实际上worker里面是while 1,不会结束)
        COORD.join(worker_threads)

        # res = np.concatenate([np.arange(len(GLOBAL_RUNNING_R)).reshape(-1, 1), np.array(GLOBAL_RUNNING_R).reshape(-1, 1)],
        #                      axis=1)
        #
        # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        # plt.xlabel('step')
        # plt.ylabel('Total moving reward')
        # plt.show()


class A3C_GAL_Worker_v1(object):
    Author = "BaoChuan Wang"
    AllowImport = True

    # 全局reward
    GLOBAL_RUNNING_R = []
    GLOBAL_EP = 0

    def __init__(self,
                 env,
                 name,
                 # 这个worker使用的本地模型
                 local_model,
                 # 用于push训练集,pull权重的worker共享模型
                 global_model,
                 # tf的sess需要全部统一,只有scope不同!
                 tf_sess,
                 save_interval=1000,
                 # 是否分析延迟,打印出延迟信息
                 delay_debug=False,
                 # A3C算法的gamma
                 gamma=0.9):
        self.env = env
        self.env.connect()
        self.name = name
        self.AC = local_model
        self.gamma = gamma
        self.sess = tf_sess
        self.env.start()
        self.global_model = global_model
        self.save_interval = save_interval
        self.delay_debug = delay_debug
        if self.delay_debug:
            self.da = DelayAnalyser()

    def work(self):

        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        current_state = self.env.reset()
        done = False

        while 1:
            # 因为下面的循环有break,所以这里的语句是done过后的重置,reset交给env,这里不需要reset
            ep_r = 0
            while 1:
                if self.delay_debug:
                    self.da.report_delay()

                # 下面这两行打开,就可以不运行RL而打印state和reward,测试其合理性
                # self.env.get_state_and_reward()
                # continue

                if self.delay_debug:
                    self.da.start("Choose Action")

                action = self.AC.choose_action(current_state)

                if self.delay_debug:
                    self.da.end("Choose Action")
                    self.da.start("Step")

                new_state, reward, done, _ = self.env.step(action)

                if self.delay_debug:
                    # 注意step过程中会sleep掉一定的时间
                    self.da.end("Step")

                ep_r += reward
                buffer_s.append(current_state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if done:  # total_step % UPDATE_GLOBAL_ITER == 0 or :
                    if self.delay_debug:
                        self.da.start("Update Model")
                    if done:
                        v_s_ = 0
                    else:
                        # 运行本地网络根据new state得到估值
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: new_state[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # ::-1是reverse操作
                        # 使用v(s) = r + v(s+1)计算target_v
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    # 更新全局网络
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 更新后从全局网络中拉下最新的权重
                    self.AC.pull_global()
                    if self.delay_debug:
                        self.da.end("Update Model")

                current_state = new_state
                total_step += 1
                if done:
                    if self.delay_debug:
                        self.da.start("Save Model")
                    if total_step % self.save_interval == 0:
                        self.global_model.save(total_step)

                    if len(self.__class__.GLOBAL_RUNNING_R) == 0:
                        # record running episode reward
                        self.__class__.GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        # 使用滑动平均来计算前100个数据均值的方法
                        self.__class__.GLOBAL_RUNNING_R.append(0.99 * self.__class__.GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", self.__class__.GLOBAL_EP,
                        "| Ep_r: %i" % self.__class__.GLOBAL_RUNNING_R[-1],
                    )
                    self.__class__.GLOBAL_EP += 1
                    if self.delay_debug:
                        self.da.end("Save Model")
                    break
