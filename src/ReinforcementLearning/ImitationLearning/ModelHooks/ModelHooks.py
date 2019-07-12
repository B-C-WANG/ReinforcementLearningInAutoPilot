# coding:utf-8
from ReinforcementLearning.Modules.Environments.Actions import ContinuousSteeringVelocityBrakeAction_v1
import numpy as np
import threading
import pygame
from ReinforcementLearning.ImitationLearning.ModelTrainers.ModelTrainers import FloatActionTrainer


class IModelHook(object):
    '''
    使用ModelHook实现模仿学习:
    正常创建RL的agent,不过提供一个hook接口,允许其他agent从env的状态中得到action
    这个action原本是RL中的模型来给出的,现在直接让成熟的算法给出,需要比RL更为合理,否则会越学越差
    于是模型就强行模仿了另一个算法
    截获state-篡改action
    '''
    def tamper_action(self,env,current_state,*args,**kwargs):
        # 模型可以任意获取输入,但是输出必须要和未hook的模型的输出兼容
        raise NotImplementedError()


class KeyBoardHook_to_throttle_or_brake_and_steering(IModelHook):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    无视state，state由人类观察，人类用键盘给出action
    keyboard hook 需要两个线程,
    一个用于获取键盘输入修改油门转向,
    另一个被调用时直接输出设置好的油门转向
    '''
    @staticmethod
    def test():
        a = KeyBoardHook_to_throttle_or_brake_and_steering()
        a.start_keyboard_control()
        while 1:
            print(a.throttle_or_brake,a.turn)

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Hello world")
        self.pygame_screen = pygame.display.set_mode((640, 480), 0, 32)
        self.throttle_or_brake = 0
        self.turn = 0

    def start_keyboard_control(self):
        # 获取键盘输入修改action将要输出的值
        def func():
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.throttle_or_brake = min(1,self.throttle_or_brake+0.1)
                        elif event.key == pygame.K_DOWN:
                            self.throttle_or_brake = max(-1, self.throttle_or_brake - 0.1)
                        elif event.key == pygame.K_LEFT:
                            self.turn = max(-1, self.turn - 0.1)
                        elif event.key == pygame.K_RIGHT:
                            self.turn = min(1, self.turn + 0.1)
                # TODO:将油门和转向在pygame窗口中显示,注意虚拟机里面有画面很卡!
        threading.Thread(target=func).start()

    def tamper_action(self,env,current_state,*args,**kwargs):
        # 输出设置好的油门/刹车和转向的2长度向量
        return np.array([self.throttle_or_brake,self.turn])




if __name__ == '__main__':
    KeyBoardHook_to_throttle_or_brake_and_steering.test()