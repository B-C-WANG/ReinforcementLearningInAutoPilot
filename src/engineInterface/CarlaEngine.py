# coding:utf-8
from engineInterface.EngineInterface import IVehicleEngine, Gear, ISmartCityEngine
import logging
from common.Math import ThreeTimesPolyFit
import sys
import threading
import random
import math
import common.Math as myMath
import warnings
import thirdParty.transformations as tf
import numpy as np
from ClosedError import ClosedError

# FIXME: 所有获得的坐标是不是车辆的base link坐标?如果不是,加一个接口修正


class CarlaSmartCityEngine(ISmartCityEngine):
    pass


class CarlaVehicleEngine(IVehicleEngine):

    def __init__(self,
                 carla_egg_path,
                 carla_pythonAPI_path):
        '''
        egg path和python api path是carla编译好pythonAPI后得到的文件，需要加入环境变量中
        :param carla_egg_path: 比如 "/XXX/carla/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg"
        :param carla_pythonAPI_path: 比如 "/XXX/carla/PythonAPI/carla"
        '''
        # 将egg path和carla的python api path加入环境变量中
        sys.path.append(carla_egg_path)
        sys.path.append(carla_pythonAPI_path)
        # 尝试导入carla
        import carla

        logging.info("If want to draw lines, make sure UE is always forced!!!")
        self.world = self.map = self.control = None
        super(CarlaVehicleEngine, self).__init__()
        self.history_esr_track_id = {}

        # 与esr相关的参数
        self.__DROP_HISTORY_TIMES = 20
        self.__CAMERA_HEIGHT = 1.26
        self.__OBSTACLE_HEIGHT = 1.50

        # 视觉和毫米波雷达的检测范围,主要是用于计算更新障碍物和esr中track的物体求距离的范围!
        self.vision_distance = 250
        self.esr_detect_distance = 150

    def connect(self, host_ip, port, time_out=4.0):
        '''
        连接在指定ip,端口启动了carlaUE或者carlaUE docker环境的server
        '''
        import carla
        self.carla_imported = carla
        self.client = carla.Client(host_ip, port)
        self.client.set_timeout(time_out)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.debug = self.world.debug
        self.control = carla.VehicleControl(
            throttle=0,
            steer=0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0
        )

        # 用于销毁自己车辆的函数
        def d():
            self.client.apply_batch([carla.command.DestroyActor(self.vehicle)])

        self.destroy_func = d

    def destroy_vehicle(self):
        '''
        在引擎中摧毁自身车辆
        '''
        self.destroy_func()

    def spawn_vehicle(self,
                      spawn_point_index=0,
                      vehicle_blueprint_name=None,
                      role_name="player"):
        '''
        根据spawn point index获得车辆生成点,然后在生成点位置生成蓝图名称为vehicle blueprint name的车辆,并赋值role name
        如果车辆生成失败(原位置已经有车了),则一直尝试下一个生成点
        '''

        # 随机获得一个车辆蓝图,可能获得自行车的蓝图,不推荐
        # blueprint = random.choice(
        #     self.world.get_blueprint_library().filter("vehicle.*")
        # )

        # 在以下蓝图中随机选择一个车辆蓝图
        if vehicle_blueprint_name is None:
            allow_car_blueprint_id = [
                "vehicle.kawasaki.ninja",
                "vehicle.lincoln.mkz2017",
                "vehicle.tesla.model3",
                "vehicle.jeep.wrangler_rubicon",
                "vehicle.bmw.isetta",
                "vehicle.bmw.grandtourer",
                "vehicle.audi.tt",
                "vehicle.ford.mustang", ]

            blueprint = self.world.get_blueprint_library().find(random.choice(allow_car_blueprint_id))
        else:
            blueprint = self.world.get_blueprint_library().find(vehicle_blueprint_name)

        total_spawn_points_num = len(self.map.get_spawn_points())
        # 设置蓝图中的指定参数
        blueprint.set_attribute("role_name", "player")

        # 根据index选择出生地点
        try:
            spawn_point = self.map.get_spawn_points()[spawn_point_index]
        except IndexError:
            warnings.warn("spawn point index is out of range, random choose one!")
            spawn_point_index = random.randint(0, total_spawn_points_num - 1)
            spawn_point = self.map.get_spawn_points()[spawn_point_index]

        # 将车辆蓝图spawn到指定位置
        self.vehicle = self.world.try_spawn_actor(
            blueprint, spawn_point
        )
        # 如果不能够生成车辆,就一直选其他点生成,循环选择所有的spawn points
        while 1:
            if self.vehicle is not None: break
            self.vehicle = self.world.try_spawn_actor(
                blueprint, self.map.get_spawn_points()[spawn_point_index + 1]
            )
            content = "Can not spawn Vehicle at index %s, try spawn at %s" % (spawn_point_index, spawn_point_index + 1)
            spawn_point_index += 1
            spawn_point_index = spawn_point_index % total_spawn_points_num
            warnings.warn(content)

    # 以下是车辆控制API
    def apply_vehicle_control(self,
                              throttle=None,
                              steer=None,
                              brake=None,
                              hand_brake=None,
                              reverse=None,
                              manual_gear_shift=None,
                              gear=None):
        if throttle is not None:
            self.control.throttle = throttle
        if steer is not None:
            self.control.steer = steer
        if brake is not None:
            self.control.brake = brake
        if hand_brake is not None:
            self.control.hand_brake = hand_brake
        if reverse is not None:
            self.control.reverse = reverse
        if manual_gear_shift is not None:
            self.control.manual_gear_shift = manual_gear_shift
        if gear is not None:
            self.control.gear = gear

        self.vehicle.apply_control(self.control)

    def apply_vehicle_brake(self, brake):
        self.control.brake = brake
        self.vehicle.apply_control(self.control)

    def apply_vehicle_gear(self, gear):
        self.control.gear = gear
        self.vehicle.apply_control(self.control)

    def apply_vehicle_steer(self, steer):
        self.control.steer = steer
        self.vehicle.apply_control(self.control)

    def apply_vehicle_throttle(self, throttle):
        self.control.throttle = throttle
        self.vehicle.apply_control(self.control)

    def apply_vehicle_reverse(self, is_reverse):
        self.control.reverse = is_reverse

    def get_vehicle_current_pose(self):
        l = self.vehicle.get_transform().location
        x = l.x
        y = l.y
        z = l.z
        e = self.vehicle.get_transform().rotation
        yaw = e.yaw
        roll = e.roll
        pitch = e.pitch
        quat = tf.quaternion_from_euler(math.radians(roll), math.radians(pitch), math.radians(yaw))
        ox = quat[0]
        oy = quat[1]
        oz = quat[2]
        ow = quat[3]

        return x, y, z, ox, oy, oz, ow

    @property
    def vehicle_yaw(self):
        # 注意是角度制
        return self.vehicle.get_transform().rotation.yaw

    def get_esr_tracks_and_obstacles(self):
        '''
        因为这里面两者代码很像,所以写在一起,但障碍物主要是车辆
        '''
        raise ClosedError("Code content is not open-source!")



    def get_velocity_scalar(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def get_accel_scalar(self):
        # 加速度标量
        v = self.vehicle.get_acceleration()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def __get_bounding_box(self, obstacle):
        # 图像旋转矩阵,将远小近大的图片转成平面用于判断,视觉给出
        raise ClosedError("Code content is not open-source!")

    def __maintain_history_track(self):
        '''
        障碍物一旦出现,就会持续保留一段时间,如果障碍物被发现过后消失,那么会在DROP TIMES过后消失,在这段期间如果障碍物重新被检测到,会刷新计数为0
        '''
        for key in self.history_esr_track_id:
            if self.history_esr_track_id[key] < self.__DROP_HISTORY_TIMES:
                self.history_esr_track_id[key] += 1
            else:
                del self.history_esr_track_id[key]

    def get_velocity(self):
        # 速度向量
        return self.vehicle.get_velocity()

    def get_angular_velocity(self):
        # 角速度向量
        return self.vehicle.get_angular_velocity()

    def get_steer_angle(self):
        # 第一个轮胎的steer angle,角度制
        return self.vehicle.get_physics_control().wheels[0].steer_angle

    def get_accel(self):
        # 车辆的加速度矢量
        return self.vehicle.get_acceleration()

    def get_vehicle_traffic_light_info(self):
        '''
        :return: 是否在等红绿灯,红绿灯状态(红还是绿),红绿灯和车的距离
        '''
        if self.vehicle.is_at_traffic_light():

            t = self.vehicle.get_traffic_light()
            if t is None:
                return False, None, None
            # 相对位置
            x, y = myMath.convert_point_into_relative_coordinate(
                target_xy=[t.get_location().x,
                           t.get_location().y],
                original_xy=[self.vehicle.get_location().x,
                             self.vehicle.get_location().y],
                original_yaw_radius=math.radians(self.vehicle_yaw))

            if x < 0:
                return False, None, None
            return True, self.vehicle.get_traffic_light_state(), x
        else:
            return False, None, None

    def get_center_lane_param(self):
        '''
        得到车道中心线的拟合参数
        20m,间隔2m采样,然后拟合这些点!
        :return 4长度向量c3, c2, c1, c0
        '''
        raise ClosedError("Code content is not open-source!")

    def get_left_right_center_lane(self):
        '''
        # TODO:所有的车道的点应当是以车辆坐标转换后进行拟合!
        获得车道线信息
        从车辆的起点开始,间隔2m向后获得点,总共50个点,同时绘制出来!
        '''
        raise ClosedError("Code content is not open-source!")

    # ________________ 下面是绘制的相关api ______________

    def draw_transform(self, trans, color=None, lift_time=1):
        if color is None:
            color = self.carla_imported.Color(255, 0, 0)
        yaw_in_rad = math.radians(trans.rotation.yaw)
        pitch_in_rad = math.radians(trans.rotation.pitch)
        p1 = self.carla_imported.Location(
            x=trans.location.x + math.cos(pitch_in_rad) * math.cos(yaw_in_rad),
            y=trans.location.y + math.cos(pitch_in_rad) * math.sin(yaw_in_rad),
            z=trans.location.z + math.sin(pitch_in_rad))
        self.debug.draw_arrow(trans.location, p1, thickness=0.05, arrow_size=1.0, color=color, life_time=lift_time)

    def draw_point_xyz(self, x, y, z, color=(255, 255, 0), thickness=0.1, life_time=1):
        location = self.carla_imported.Location(x=x, y=y, z=z)
        color = self.carla_imported.Color(*color)
        self.debug.draw_point(location, thickness, color,
                              life_time, False)

    def draw_waypoint_list(self, waypoint_list, color=None, thickness=0.05, life_time=1, name="waypoints"):
        if len(waypoint_list) == 0: return
        if color is None:
            color = self.carla_imported.Color(255, 0, 0)
        else:
            color = self.carla_imported.Color(*color)
        last_waypoint = waypoint_list[0]
        # TODO:debug:没有绘制出文字!
        # self.debug.draw_string(waypoint_list[0].transform.location + self.carla_imported.Location(z=1), "12312321321", False, self.carla_imported.Color(0,255,255), life_time=life_time)
        for i in range(1, len(waypoint_list)):
            next_waypoint = waypoint_list[i]
            # self.debug.draw_line(
            #    last_waypoint.transform.location + self.carla_imported.Location(z=0.25),
            #    next_waypoint.transform.location + self.carla_imported.Location(z=0.25),
            #    thickness=0.1, color=color, life_time=life_time, persistent_lines=False
            # )
            self.debug.draw_point(last_waypoint.transform.location + self.carla_imported.Location(z=0.25), thickness,
                                  color, life_time, False)
            last_waypoint = next_waypoint
        self.debug.draw_point(last_waypoint.transform.location + self.carla_imported.Location(z=0.25), thickness, color,
                              life_time, False)

    def make_transform(self, x=0, y=0, z=0, yaw=0, roll=0, pitch=0):
        t = self.carla_imported.Transform()
        t.location = self.carla_imported.Location(x=x, y=y, z=z)
        t.rotation = self.carla_imported.Rotation(yaw=yaw, roll=roll, pitch=pitch)
        return t

    def draw_line_location(self, s, e, thickness=0.3, color=(255, 0, 0), life_time=1, persistent_lines=False):
        color = self.carla_imported.Color(*color)

        self.debug.draw_line(
            s, e,
            thickness=0.3,
            color=color,
            life_time=life_time,
            persistent_lines=False
        )

    def draw_line(self, s, e, thickness=0.3, color=(255, 0, 0), life_time=1, persistent_lines=False):
        s = s.transform.location + self.carla_imported.Location(z=0.25)
        e = e.transform.location + self.carla_imported.Location(z=0.25)
        color = self.carla_imported.Color(*color)
        self.debug.draw_line(
            s, e,
            thickness=0.3,
            color=color,
            life_time=life_time,
            persistent_lines=False
        )

    def draw_3deg_lane(self, params, original_point, life_time=0.2, color=None):
        '''
        :param params: 车道线拟合参数,低次项-高次项
        :param length: 车道线总长
        :param original_point: 车的世界坐标(用于转曲线的世界坐标)
        :return:
        算法:
        这个曲线一定是以车的方向为x的,所以只需要将x等于一系列的值代入,得到y值,然后加上车的坐标
        转到世界坐标进行绘制!
        '''
        if color is None:
            color = self.carla_imported.Color(0, 0, 255)
        assert len(params) == 4, "Only 4 params in 3 deg polyfit!"
        x = np.linspace(0, 10, 100)
        y = np.power(x, 3) * params[3] + np.power(x, 2) * params[2] + x * params[1] + params[0]
        line_points = []
        for i in range(x.shape[0]):
            world_pos = myMath.convert_point_info_world_coordinate(
                target_xy=[y[i], x[i]],
                original_xy=[original_point[0], original_point[1]],
                original_yaw_radius=math.radians(self.vehicle_yaw)
            )
            line_points.append(
                self.carla_imported.Location(x=world_pos[0], y=world_pos[1], z=0.25))

        last_point = line_points[0]
        # print("Point")
        # print(x[0],y[0])
        # print(self.vehicle.get_transform().location)
        # print(last_point)

        for i in range(1, len(line_points)):
            next_point = line_points[i]
            self.debug.draw_line(
                last_point,
                next_point,
                thickness=0.3,
                color=color,
                life_time=life_time,
                persistent_lines=False
            )
            last_point = next_point
