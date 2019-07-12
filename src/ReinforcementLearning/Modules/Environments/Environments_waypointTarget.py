# coding:utf-8
# Type: MultiAuthor

from ReinforcementLearning.Modules.Environments.Actions import SimpleSteeringAction_v1, HighPrecisionControl_v1
from ReinforcementLearning.Modules.Environments.States import MoveTargetAndCarState_v1, \
    CenterLaneThreeTimesPolyFitParamState_v1
from ReinforcementLearning.Modules.Environments.Done import DistanceToTargetLowestStepAndLowVelDone_v1
from engineInterface.CarlaEngine import CarlaVehicleEngine
from ReinforcementLearning.Modules.Environments.Rewards import CarToPathDistanceDeltaReward_v1, \
    CarToPathAndTargetWaypointDistanceReward_v1, \
    OnlyCarToTargetWaypointDistanceRatioReward_v1
from ReinforcementLearning.Modules.carlaUtils import CarlrUtils as  cu
import numpy as np
import time
from ReinforcementLearning.Modules.Environments.IEnv import IEnv


class CarlaConsecutiveWaypointsTargetEnv_v1(IEnv):
    Author = "BaoChuan Wang"
    AllowImport = True

    '''
    默认env的描述:
    使用waypoint作为target的连续RL环境,
    车辆在一开始获得engine给出的waypoints(后面可以提供自定的waypoint输入),然后车辆每次只会有上一个WayPoint和下一个waypoint的信息,
    这两个waypoint会得到前进方向,这个前进方向向量会用于求得车辆到前进方向的横向距离,和速度的夹角,和加速度夹角等state
    reward是由下一个waypoint距离,和车道中心偏差,转向等求得(可替换)
    action可以是离散的左转右转,也可以是连续的油门转向(可替换)
    当车辆到达下一个waypoint时(距离下一个waypoint的比例小于路程总比例的20%),视为完成一个任务,为done,但是此时不会reset车辆位置,而是更新下一个waypoint,然后视作新的任务开始
    只有当车辆失败的时候,也就是在速度较低,或者车辆以及越过waypoint但是没有到达waypoint的时候,会视为失败并reset,此时车辆会重置
    
    因为waypoint是采用相邻两个点,因此化曲为直,对于弯道给的信息可能不够
    
           
    '''

    def __init__(self,
                 carla_egg_path="/home/wang/Desktop/carla/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg",
                 carla_pythonAPI_path="/home/wang/Desktop/carla/PythonAPI/carla",
                 carla_UE_ip="10.10.9.128",
                 carla_UE_port=2000,
                 # 多少个路径点
                 n_waypoint=60,
                 # 每个路径点间隔
                 waypoint_spacing=10,
                 # 车辆起点的index,和位置对应关系不明
                 vehicle_start_point_index=1,
                 # 视为达到target的比例,车辆到target距离/总路程小于这个值就视作成功到达
                 ratio_of_reaching=0.15,
                 # 在step函数中,应用了之前的action过后的等待时间(等待响应reward和New state!)
                 wait_time_after_apply_action=0.1,
                 # 是否增量添加车道线的拟合参数信息?
                 add_center_lane_state=False,
                 # 使用新的action替换原有默认的action(SimpleSteeringAction_v1)
                 action_replace=None,
                 # 使用新的reward去代替原有默认的reward,注意,reward的输入参数需要兼容原来的
                 reward_replace=None):
        self.n_waypoint = n_waypoint
        self.waypoint_spacing = waypoint_spacing
        self.vehicle_start_point_index = vehicle_start_point_index
        self.wait_time_after_apply_action = wait_time_after_apply_action

        # 设置所有的模块,这里这样采用是为了便于快速知晓模型的所有参数!
        # 可观测到的state
        self.move_target_and_car_state = MoveTargetAndCarState_v1()
        if add_center_lane_state:
            self.center_lane_state = CenterLaneThreeTimesPolyFitParamState_v1()
        else:
            self.center_lane_state = None
        if action_replace is not None:
            self.action = action_replace
        else:
            self.action = SimpleSteeringAction_v1(steer_big=0.5, steer_small=0.25)

        # reward采用和path的distance
        # self.reward = CarToPathDistanceReward(max_speed=300, min_speed=10, threshold_distance=3.5,beta=3)
        # self.reward = CarToPathDistanceDeltaReward_v1()
        # self.reward = CarToPathAndTargetWaypointDistanceReward_v1()
        if reward_replace is None:
            self.reward = OnlyCarToTargetWaypointDistanceRatioReward_v1()
        else:
            self.reward = reward_replace

        self.done_condition = DistanceToTargetLowestStepAndLowVelDone_v1(ratio_of_reaching=ratio_of_reaching,
                                                                         minimum_action_taken_on_low_velocity=50)
        # 当前起点和终点(指的是在一次RL中)
        self.start_waypoint_xy_array = None
        self.target_waypoint_xy_array = None

        # 最终起点和终点(起点是reset车辆到达的点,终点是done了过后需要reset的点)
        self.init_waypoint_xy_array = None
        self.final_waypoint_xy_array = None

        self.carla_egg_path = carla_egg_path
        self.carla_pythonAPI_path = carla_pythonAPI_path
        self.carla_UE_ip = carla_UE_ip
        self.carla_UE_port = carla_UE_port
        self.engine = None

    def connect(self):
        self.engine = CarlaVehicleEngine(
            carla_egg_path=self.carla_egg_path,
            carla_pythonAPI_path=self.carla_pythonAPI_path,
        )
        self.engine.connect(host_ip=self.carla_UE_ip, port=self.carla_UE_port)

    def start(self):

        if self.engine is None:
            raise ValueError("Engine is None, have you run 'connect()' ? ")
        # 车辆起点的生成地点
        self.reset_spawn_point = self.vehicle_start_point_index

        # 固定使用mkz作为车辆,如果不使用固定车辆,会造成state和action的一对多,同样的state因为车辆不同导致action不同
        self.vehicle_blueprint_name = "vehicle.lincoln.mkz2017"
        self.engine.spawn_vehicle(spawn_point_index=self.reset_spawn_point,
                                  vehicle_blueprint_name=self.vehicle_blueprint_name)
        # 生成车辆之后等一会儿再绘制
        time.sleep(1)
        self.now_vehicle_runned_step = 0

        # 车辆在每个步骤中的和路径的夹角累计绝对值差异(reward计算依据)
        self.sum_of_abs_car_to_way_angle_diff = 0
        self.now_car_to_way_angle_diff = 0

        # 初始就决定车辆要走的路径点
        # 一共15个点,每个间隔5m,只需要每个点的xy坐标,以及最后点的xy坐标
        _, _, self.car_waypoints_xy_array, self.target_waypoint_xy_array = cu.get_car_target_waypoints(
            engine=self.engine,
            vehicle=self.engine.vehicle,
            n_waypoint=self.n_waypoint,
            # 注意间隔不能太小,因为有到终点的计算延迟!
            waypoint_spacing=self.waypoint_spacing)

        # 目标点的index
        self.final_waypoint_index = self.car_waypoints_xy_array.shape[0] - 1
        self.index_of_waypoint_vehicle_on = 0  # 表明起始车辆在从0到1这个waypoint路途中!
        self.car_to_target_distance_ratio = 0
        self.init_waypoint_xy_array = self.car_waypoints_xy_array[0, :]
        self.final_waypoint_xy_array = self.car_waypoints_xy_array[-1, :]
        self.target_xy_array_relate_to_car = None

    def update_start_and_target_waypoint(self):
        '''
        在每一个小的RL任务中,根据车辆完成的WayPoint的index来更新start和target点!
        '''
        self.start_waypoint_xy_array = self.car_waypoints_xy_array[self.index_of_waypoint_vehicle_on, :]
        self.target_waypoint_xy_array = self.car_waypoints_xy_array[self.index_of_waypoint_vehicle_on + 1, :]

    def reset(self):
        # 重置起点
        self.index_of_waypoint_vehicle_on = 0
        self.update_start_and_target_waypoint()
        self.car_to_target_distance_ratio = 0
        self.engine.destroy_vehicle()
        self.engine.spawn_vehicle(spawn_point_index=self.reset_spawn_point,
                                  vehicle_blueprint_name=self.vehicle_blueprint_name)
        # 等待车辆稳住
        time.sleep(1)
        self.now_vehicle_runned_step = 0
        self.sum_of_abs_car_to_way_angle_diff = 0
        self.reward.reset()
        self.action.reset()
        self.steer_of_action = 0
        state, reward = self.get_state_and_reward()
        print("------Env Reset------")
        return state

    @property
    def observation_space(self):
        space = self.move_target_and_car_state.space_shape[0]
        if self.center_lane_state is not None:
            space += self.center_lane_state.space_shape[0]
        return (space,)

    @property
    def action_space(self):
        return self.action.space_shape

    def step(self, action):
        data = self.action.interpret_action(action)
        throttle = data["throttle"]
        brake = data["brake"]
        steering = data["steering"]
        self.steer_of_action = steering
        self.engine.apply_vehicle_control(
            brake=brake,
            steer=steering,
            throttle=throttle
        )
        time.sleep(self.wait_time_after_apply_action)
        self.now_vehicle_runned_step += 1
        # 积累角度差作为cost,避免蛇形走位
        self.sum_of_abs_car_to_way_angle_diff += abs(self.now_car_to_way_angle_diff)
        state, reward = self.get_state_and_reward()
        done, reward = self.done_condition.is_done(
            target_xy_array_relate_to_car=self.target_xy_array_relate_to_car,
            car_to_target_distance_ratio=self.car_to_target_distance_ratio,
            now_vehicle_runned_step=self.now_vehicle_runned_step,
            velocity=self.engine.get_velocity_scalar(),
            now_reward=reward,
            sum_of_abs_car_to_way_angle_diff=self.sum_of_abs_car_to_way_angle_diff
        )
        # 对done进行再次处理,因为最后给出的都是Bool
        if done == 2:
            # print("sum of abs car to way angle: ",self.sum_of_abs_car_to_way_angle_diff)
            # 重置car to way夹角之和
            self.sum_of_abs_car_to_way_angle_diff = 0
            if self.index_of_waypoint_vehicle_on + 1 >= self.final_waypoint_index:
                print("Pass All waypoints! Mission Complete!")
                self.reset()
                done = "AllDone"
                done = True
            else:
                print("Pass waypoint %s success!" % self.index_of_waypoint_vehicle_on)
                # 如果只是通过了另一个点,就只需要更新下一个点即可
                self.index_of_waypoint_vehicle_on += 1
                self.update_start_and_target_waypoint()
                # 这个done不会用于reset,因为reset是env控制,这里的done只是给模型提示说需要进行一批次训练了!
                done = "PartDone"
                done = True

        elif done == 1:
            self.reset()
            done = "FailDone"
            done = True
        elif done == 0:
            done = False
        # print("done: ",done)

        return state, reward, done, action

    def get_state_and_reward(self):
        self.update_start_and_target_waypoint()
        # 路径点state
        way_unit_direction, car_to_way_unit_direction, car_forward_unit_direction, self.car_to_target_distance_ratio, self.target_xy_array_relate_to_car = \
            cu.get_direction_vector_series_and_car_to_next_waypoint_ratio(self.engine,
                                                                          self.start_waypoint_xy_array,
                                                                          self.target_waypoint_xy_array)
        velocity_vector = self.engine.get_velocity()
        velocity_vector = np.array([-velocity_vector.x, -velocity_vector.y])
        accel_vector = self.engine.get_accel()
        accel_vector = np.array([-accel_vector.x, -accel_vector.y])

        state_array, state_info_dict = self.move_target_and_car_state.merge_states(
            target_xy_array_relate_to_car=self.target_xy_array_relate_to_car,
            way_unit_direction_vector=way_unit_direction,
            car_to_way_unit_direction_vector=car_to_way_unit_direction,
            car_forward_unit_direction_vector=car_forward_unit_direction,
            velocity_direction_vector=velocity_vector,
            accel_direction_vector=accel_vector)

        reward = self.reward.get_reward(
            car_velocity=state_info_dict["velocity"],
            steer_of_action=self.steer_of_action,
            car_to_target_distance_ratio=self.car_to_target_distance_ratio,
            target_y_relate_to_car=self.target_xy_array_relate_to_car[1])

        # 合并车道线state,如果有的话
        if self.center_lane_state is not None:
            center_lane_param = self.engine.get_center_lane_param()
            state_array_to_append = self.center_lane_state.merge_states(center_lane_param)
            state_array = np.concatenate([state_array, state_array_to_append], axis=0)

        self.now_car_to_way_angle_diff = state_info_dict["car_to_way_angle_diff"]
        return state_array, reward