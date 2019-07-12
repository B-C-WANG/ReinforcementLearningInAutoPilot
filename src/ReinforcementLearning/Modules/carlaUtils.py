# coding:utf-8
# Type: Public

import numpy as np
import common.Math as  cMath
import math


class CarlrUtils(object):
    Author = "BaoChuan Wang"
    AllowImport = False

    @staticmethod
    def get_direction_vector_series_and_car_to_next_waypoint_ratio(
            carla_engine,
            start_waypoint_xy_array,
            target_waypoint_xy_array,
            draw_in_UE=False
    ):
        '''
        适用于WaypointsTarget环境的state求取


        # 以下代码作为参考
        获得车辆最近的路径点,以及接下来n个路径点(目前改为最后两个路径点,不会随着车辆位置更新!),然后返回与这两个路径点相关的参数,有:
        1.车辆到两个waypoints的中点距离
        2.waypoint方向角
        3.车辆到waypoint中点方向角
        4.车辆本身方向角

        #  另外这样获取waypoints实时更新的方法是不合适的,产生的rewards不对action连续

        # 原来的方法是车辆获取最近的waypoint然后求得下一个waypoints,现在改成一开始就确定waypoints
        因为使用获取最近waypoints的方法可能会产生变道
        原来的方法代码:
        # # 获得车辆的下两个waypoints的xy坐标
        # next_center_waypoints = self.engine.map.get_waypoint(
        #     # location
        #     self.engine.vehicle.get_location()
        # )
        # # 获得接下来5m的作为下一个路径点
        # next_next_center_waypoints = next_center_waypoints.next(5)[0]
        #
        # waypoint_list =((
        #            next_center_waypoints.transform.location.x,
        #            next_center_waypoints.transform.location.y
        #        ), (
        #            next_next_center_waypoints.transform.location.x,
        #            next_next_center_waypoints.transform.location.y
        #        ))
        #
        # # 在carla中绘制路径点
        # self.engine.draw_waypoint_list(
        #     [next_center_waypoints,next_next_center_waypoints],life_time=1)
        #
        # return waypoint_list

        # 注意点:
        因为最终计算的时候是需要两个waypoint来得到和车辆的距离
        以及 车辆到waypoints中心点的方向 和 两个waypoints方向 的夹角
        所以一定要保证waypoints中心点在车辆前方(否则就会后退)
        需要保证Waypoints的间隔足够大即可!也可以这里取点时取后面两个点而不是一个点!

        # 这里的代码是求得距离车辆最近的点,然后往下找3个点,现在更新成一开始指定的点!
        # # 求得最近的waypoints的index,然后找下一个!如果到了waypoints的末端?
        # distance = np.sqrt(
        #     np.sum(np.square(self.car_waypoints_xy_array - np.array([self.engine.vehicle.get_location().x,
        #                                                              self.engine.vehicle.get_location().y])), axis=1))
        #
        # # print(distance)
        # # 最大的index
        # index_max = distance.shape[0] - 1
        # # 找到距离最近的waypoints的index
        # index = int(np.argmin(distance))
        #
        #
        # index = index_max - 1
        #
        # # 这里点取得向前一点儿
        # next_point_index = index + 3
        # if next_point_index > index_max: next_point_index = index_max

        # if draw_in_UE:
        #     # 作出两个waypoints的线段
        #     start = self.car_waypoints_list[index]
        #     end = self.car_waypoints_list[next_point_index]
        #     self.engine.draw_line(start, end, life_time=1, color=(0, 255, 0))
        # center_point = (self.car_waypoints_xy_array[index, :].reshape(-1) +
        #                 self.car_waypoints_xy_array[next_point_index, :].reshape(-1)) / 2

        '''

        # 车辆位置
        vehicle_location = carla_engine.vehicle.get_location()
        car_point = np.array([vehicle_location.x, vehicle_location.y])

        if draw_in_UE:
            # waypoint中点
            center_point = (start_waypoint_xy_array + target_waypoint_xy_array) / 2
            center_point_transform = carla_engine.make_transform(
                x=center_point[0],
                y=center_point[1],
                z=vehicle_location.z
            )
            carla_engine.draw_point_xyz(center_point[0], center_point[1], carla_engine.vehicle.get_location().z + 0.25,
                                        color=(0, 255, 255), thickness=0.1)

            carla_engine.draw_line_location(
                vehicle_location,
                center_point_transform.location,
                life_time=1, color=(0, 0, 255)
            )

        # waypoints的单位方向向量
        way_unit_direction = target_waypoint_xy_array - start_waypoint_xy_array
        way_unit_direction /= np.linalg.norm(way_unit_direction, 2)
        # 车辆到中心点的单位方向向量
        car_to_way_unit_direction = (target_waypoint_xy_array - car_point)
        car_to_way_unit_direction /= np.linalg.norm(car_to_way_unit_direction, 2)

        # 车辆本身的单位方向向量
        car_unit_direction = carla_engine.vehicle.get_transform().get_forward_vector()
        car_unit_direction = np.array([car_unit_direction.x, car_unit_direction.y])

        # 车辆到target点和总路程的比值
        total_distance = np.linalg.norm(target_waypoint_xy_array - start_waypoint_xy_array, 2)
        now_distance = np.linalg.norm(target_waypoint_xy_array - car_point, 2)
        car_to_target_distance_ratio = now_distance / total_distance

        # 车辆的yaw角度
        car_yaw = math.radians(carla_engine.vehicle_yaw)

        # 增加:相对于车辆坐标的目标waypoint的x和y
        target_xy_array_relate_to_car = cMath.convert_point_into_relative_coordinate(
            target_waypoint_xy_array,
            car_point,
            original_yaw_radius=car_yaw)

        return way_unit_direction, car_to_way_unit_direction, car_unit_direction, car_to_target_distance_ratio, target_xy_array_relate_to_car

    @staticmethod
    def get_car_target_waypoints(engine, vehicle, n_waypoint=2, waypoint_spacing=15, draw_waypoints=True):

        if n_waypoint < 2:
            raise ValueError("At least 2 waypoints will return!")

        # List<Waypoints>
        car_waypoints_list = []
        # Array2D
        car_waypoints_xy_array = None
        # List<List>
        car_waypoints_xy_list = []

        # 起始的点
        next_center_waypoints = engine.map.get_waypoint(vehicle.get_location())
        # 车辆的起点
        start_waypoint_xy_array = np.array([next_center_waypoints.transform.location.x,
                                            next_center_waypoints.transform.location.y])
        car_waypoints_list.append(next_center_waypoints)
        car_waypoints_xy_list.append([next_center_waypoints.transform.location.x,
                                      next_center_waypoints.transform.location.y])
        if n_waypoint == 2:
            next_center_waypoints = next_center_waypoints.next(waypoint_spacing)[0]
            car_waypoints_list.append(next_center_waypoints)
            car_waypoints_xy_list.append([next_center_waypoints.transform.location.x,
                                          next_center_waypoints.transform.location.y])
        else:
            for i in range(n_waypoint - 1):
                next_center_waypoints = next_center_waypoints.next(waypoint_spacing)[0]
                car_waypoints_list.append(next_center_waypoints)
                car_waypoints_xy_list.append([next_center_waypoints.transform.location.x,
                                              next_center_waypoints.transform.location.y])
        car_waypoints_xy_array = np.array(car_waypoints_xy_list)
        # 终点
        target_waypoint_xy_array = np.array([next_center_waypoints.transform.location.x,
                                             next_center_waypoints.transform.location.y])
        # 绘制路径点
        if draw_waypoints:
            engine.draw_waypoint_list(car_waypoints_list, life_time=99999)

        return car_waypoints_list, car_waypoints_xy_list, car_waypoints_xy_array, target_waypoint_xy_array

    @staticmethod
    def get_velocity_accel_relative_to_car_and_their_scalar(engine):
        velocity_vector = engine.get_velocity()
        velocity_to_car_x, velocity_to_car_y = cMath.convert_point_into_relative_coordinate(
            target_xy=[velocity_vector.x, velocity_vector.y],
            original_xy=[0, 0],
            original_yaw_radius=math.radians(engine.vehicle_yaw))

        velocity = engine.get_velocity_scalar()

        accel_vector = engine.get_accel()
        accel_to_car_x, accel_to_car_y = cMath.convert_point_into_relative_coordinate(
            target_xy=[accel_vector.x, accel_vector.y],
            original_xy=[0, 0],
            original_yaw_radius=math.radians(engine.vehicle_yaw))

        accel = engine.get_velocity_scalar()

        return velocity, velocity_to_car_x, velocity_to_car_y, accel, accel_to_car_x, accel_to_car_y
