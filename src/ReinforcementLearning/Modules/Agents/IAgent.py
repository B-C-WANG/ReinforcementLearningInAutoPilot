# coding:utf-8
class IAgent(object):
    '''
    负责调用Model和env，控制训练流程
    '''
    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

