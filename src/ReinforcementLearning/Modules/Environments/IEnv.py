# coding:utf-8
class IEnv(object):
    # 尽可能让IEnv的接口和gym.env保持一致,以兼容现有RL算法

    @property
    def observation_space(self):
        raise NotImplementedError()
        # return tuple like (4,)

    @property
    def action_space(self):
        raise NotImplementedError()
        # return tuple like (4,)

    def reset(self):
        raise NotImplementedError()
        # return a state vector

    def step(self, action):
        raise NotImplementedError()
        # return next_state, reward, done, action(itself)

    def start(self):
        raise NotImplementedError()

    def connect(self,*args,**kwargs):
        # 连接到engine
        raise NotImplementedError()