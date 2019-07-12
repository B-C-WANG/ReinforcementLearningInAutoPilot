# coding:utf-8

'''
EngineInterface放置所有仿真器引擎调用的接口
仿真器Carla，AirSim，carmarker等需要实现这里的接口
外部调用时无需知道引擎内部的实现，只需要调用这里面的接口，实现多个引擎仿真的统一

python的接口特殊之处在于，不要求继承接口的类的参数和接口一致,但是需要完全包括接口中的所有参数，多余的参数必须要有合适的默认值

TODO: 目前只有一个carlaEngine实现接口,因此接口没有统一规定
'''

class Gear(object):
    Park = "Park"
    Drive = "Drive"
    # TODO:完成其他档位


class ISmartCityEngine(object):
    '''
    智慧城市信息接口,多为静态函数,提供车辆之间的信息
    '''

class IVehicleEngine(object):
    '''
    车辆信息接口
    '''
    def __init__(self):
        self.client = None
        self.vehicle = None

    def connect(self,host_ip,port,time_out):
        '''
        连接到仿真器引擎中
        '''
        raise NotImplementedError()

    def spawn_vehicle(self):
        raise NotImplementedError()

    def apply_vehicle_throttle(self,throttle):
        raise NotImplementedError()

    def apply_vehicle_steer(self,steer):
        raise  NotImplementedError()

    def apply_vehicle_brake(self,brake):
        raise NotImplementedError()

    def apply_vehicle_gear(self,gear):
        raise NotImplementedError()

    def apply_vehicle_control(self,
                              throttle,
                              steer,
                              brake,
                              hand_brake,
                              reverse,
                              manual_gear_shift,
                              gear):
        raise NotImplementedError()

