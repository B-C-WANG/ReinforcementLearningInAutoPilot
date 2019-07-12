# coding:utf-8
import numpy as np
import math


def PointEuclideanDistance(point1, point2):
    return np.sqrt(np.sum([(point1.x - point2.x) ** 2, (point1.y - point2.y) ** 2]))


def EuclideanDistanceOfTwoArray(array1, array2):
    return np.sqrt(np.sum(np.square(array1 - array2)))


def PointManhattenDistance(point1, point2):
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)


def AngleOfVectorAB(pointA, pointB):
    # 求得ab点向量
    x = pointB.x - pointA.x
    y = pointB.y - pointA.y
    return math.atan2(y, x)


def normalize_angle(angle):
    a =  math.fmod(math.fmod(angle,2.0*math.pi)+2.0*math.pi,2.0*math.pi)
    if (a > math.pi ):
        a -= 2.0 * math.pi
    return a

def get_yaw_angle(x,y):
    return normalize_angle(math.atan2(y,x))



def convert_point_info_world_coordinate(target_xy,original_xy,original_yaw_radius):
    dist = math.sqrt( math.pow(target_xy[0],2) + math.pow(target_xy[1],2))
    diff_angle = get_yaw_angle(target_xy[0],target_xy[1])
    angle = normalize_angle(diff_angle+original_yaw_radius)
    diff_x = dist * math.cos(angle)
    diff_y = dist * math.sin(angle)
    return [diff_x+original_xy[0],diff_y+original_xy[1]]


def convert_point_into_relative_coordinate(target_xy, original_xy, original_yaw_radius):
    # /**
    #  * @brief transfer pose in world coordinate to relative coordinate
    #  * @param pose_in_world: point of the object in world coordinate
    #  * @param origin_pose: current pose of the relative coordinate (z is the yaw
    #  * angle)
    #  * @return point in relative coordinate
    #  */

    diff_x = target_xy[0] - original_xy[0]
    diff_y = target_xy[1] - original_xy[1]
    cosr = math.cos(original_yaw_radius)
    sinr = math.sin(original_yaw_radius)

    return diff_x * cosr + diff_y * sinr, diff_y * cosr - diff_x * sinr


def ThreeTimesPolyFit(point_list):
    '''
    输入是一系列的点[(x1,y1),(x2,y2)]
    '''
    x = []
    y = []
    for i in point_list:
        x.append(i[0])
        y.append(i[1])
    # 拟合结果参数是高次在前
    c3, c2, c1, c0 = list(np.polyfit(x, y, 3))
    return c3, c2, c1, c0

