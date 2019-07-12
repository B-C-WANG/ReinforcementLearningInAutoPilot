# coding:utf-8
# Type: MultiAuthor

from ReinforcementLearning.Modules.utils import LinearEpsilonAnnealingExplorer

class ITPManager(object):
    '''
    用于给agent提供训练流程控制的Flag
    '''
    def should_use_random_action(self):
        # 使用随机决策还是模型决策?随机决策为True
        raise NotImplementedError()

    def should_train(self):
        # 是否模型训练
        raise NotImplementedError()

    def should_save_model(self):
        # 是否保存模型
        raise NotImplementedError()


class PretrainLinearEGreedyAnnealingManager(ITPManager):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    几个阶段
    1. 预训练:此时完全随机
    2. e-greedy退火:按照概率随机,随机的概率线性下降
    3. 过了预训练阶段之后间隔train interval开始训练,积累训练步数
    4. 间隔一段训练步数过后开始存储权重

    '''

    def __init__(self,
                 e_greedy_start=1,
                 e_greedy_end=0.1,
                 e_greedy_total_steps=100000,
                 n_pretrain_steps=20000,
                 n_train_interval_steps=4,
                 n_model_save_interval=1000):
        # 线性e-greedy退火,一定概率随机选择action,一定概率选择q值最高的action,选择随机的概率持续减小
        self.__explorer = LinearEpsilonAnnealingExplorer(e_greedy_start, e_greedy_end, e_greedy_total_steps)

        # 预热训练步数(不使用RL的随机action)
        # 调整使用NN进行预测的预热步数,因为有退火e-greedy,所以可以设置低一点儿
        self.__n_pretrain_steps = n_pretrain_steps

        # 每间隔steps进行训练
        self.__n_train_interval_steps = n_train_interval_steps

        # 每训练多少次之后存储模型
        self.__n_model_save_interval = n_model_save_interval

        # public:用于调用来增加步数

        # 总共action了多少步
        self.__steps_action_taken = 0
        # 训练了多少步
        self.__steps_train_taken = 0

    @property
    def steps_train_taken(self):
        return self.__steps_train_taken

    @property
    def steps_action_taken(self):
        return self.__steps_action_taken

    def train_step_plus(self):
        self.__steps_train_taken += 1

    def action_step_plus(self):
        self.__steps_action_taken += 1

    def should_use_random_action(self):
        return self.__explorer.is_exploring(self.__steps_action_taken)

    def should_train(self):
        if self.__steps_action_taken > self.__n_pretrain_steps and \
                self.__steps_action_taken % self.__n_train_interval_steps == 0:
            return True
        return False

    def should_save_model(self):
        if self.__steps_train_taken > 0 and self.__steps_train_taken % self.__n_model_save_interval == 0:
            return True
        return False

