import math
from typing import *

import numpy as np

import utils


class Line(object):
    """
    """

    def __init__(self, start: np.ndarray, end: np.ndarray):
        """
        """
        self.start = start
        self.end = end

    @property
    def direction(self):
        """

        :return:
        """
        return utils.normalize(self.end - self.start)

    
    def intersection(self, other: Union['Line', 'Plane']):
        """

        :param other:
        :return:
        """
        if isinstance(other, Line):
            return line_intersection_line(self, other)
        elif isinstance(other, Plane):
            return other.intersection(self)
        else:
            raise AttributeError(f'Illegal parameter type: {type(other)}, should be Line or Plane.')


class Plane(object):
    """

    """

    def __init__(self, center: np.ndarray, normal_direction: np.ndarray):
        """
        """
        self.center = center
        normal_direction = utils.normalize(normal_direction)
        self.normal_direction = normal_direction

    def intersection(self, other: Line):
        """

        :param other:
        :return:
        """
        plane_point = self.center
        normal_vector = self.normal_direction
        if isinstance(other, Line):
            return plane_intersection_line(plane_point, normal_vector, other)


class Sphere(object):
    """

    """

    def __init__(self,
                 center: np.ndarray,
                 radius: float,
                 refractive_rate: float,
                 surface_color: np.ndarray,
                 emission_color: Union[np.ndarray, None] = None,
                 reflection: float = 0.,
                 transparency: float = 0.):
        """

        :param center:
        :param radius:
        :param surface_color:
        :param emission_color:
        :param transparency:
        :param reflection:
        """
        self.center = center
        self.radius = radius
        self.refractive_rate = refractive_rate
        self.surface_color = surface_color
        if emission_color is None:
            self.emission_color = np.array([0., 0., 0.], dtype=np.float64)
        else:
            self.emission_color = emission_color

        self.transparency = transparency
        self.reflection = reflection

    def intersection(self,
                     source_point: np.ndarray,
                     ray_direction: np.ndarray) -> Tuple[Union[None, np.ndarray], float]:
        """

        :param source_point:
        :param ray_direction:
        :return:
        """
        ray = self.center - source_point
        projection = np.asscalar(np.dot(ray, ray_direction))
        if projection < 0:
            # 方向不对
            return None, float('inf')
        perpendicular_vec = ray - projection * ray_direction
        length = self.radius * self.radius - utils.square_norm(perpendicular_vec)
        if length < 0:
            # 未相交
            return None, float('inf')
        length = math.sqrt(length)
        scale_0 = projection - length
        scale_1 = projection + length
        if scale_0 < 0:
            return scale_1 * ray_direction + source_point, scale_1
        else:
            return scale_0 * ray_direction + source_point, scale_0


def line_intersection_line(line_1: Line, line_2: Line):
    """
    """
    a_1, b_1, c_1 = line_1.start
    a_2, b_2, c_2 = line_1.end

    a_3, b_3, c_3 = line_2.start
    a_4, b_4, c_4 = line_2.end

    coeff_mat = np.mat([[b_2 - b_1, -(a_2 - a_1), 0],
                        [c_2 - c_1, 0, -(a_2 - a_1)],
                        [b_4 - b_3, -(a_4 - a_3), 0],
                        [c_4 - c_3, 0, -(a_4 - a_3)]])
    rhs = np.array([a_1 * b_2 - a_2 * b_1,
                    a_1 * c_2 - a_2 * c_1,
                    a_3 * b_4 - a_4 * b_3,
                    a_3 * c_4 - a_4 * c_3]).reshape(4, 1)
    if np.linalg.matrix_rank(coeff_mat) < 4:
        try:
            res_mat, residual, _, _ = np.linalg.lstsq(coeff_mat, rhs)
        except np.linalg.LinAlgError:
            return None
        else:
            if abs(residual) < utils.SMALL_EPSILON:
                return res_mat.reshape(3, )
            else:
                return None
    else:
        return None


def plane_intersection_line(plane_point: np.ndarray,
                            normal_vec: np.ndarray,
                            line: Line):
    """
    """
    ray_source = line.start
    ray_direction = line.direction
    if abs(np.dot(normal_vec, ray_direction)) < 1e-6:
        return None
    ray = plane_point - ray_source
    projection = np.asscalar(np.dot(ray, normal_vec))
    if projection < 0:
        normal_vec = - normal_vec

    projection_point = abs(projection) * normal_vec + ray_source
    print(projection_point)
    projection_direction = ray_direction - np.dot(ray_direction, normal_vec) * normal_vec
    print(projection_direction)
    projection_line = Line(projection_point, projection_direction + projection_point)
    return line_intersection_line(line, projection_line)
