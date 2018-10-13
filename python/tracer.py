import concurrent.futures
import logging
import math
import time
from typing import *

import numpy as np

import geometric
import utils

logging.basicConfig(level=logging.INFO)

MAX_DEPTH = 5


def mix(a: float, b: float, mix_rate: float) -> float:
    """

    :param a:
    :param b:
    :param mix_rate:
    :return:
    """
    return a * (1 - mix_rate) + b * mix_rate


def ray_trace(ray_source: np.ndarray, ray_direction: np.ndarray, sphere_list: List[geometric.Sphere], depth: int):
    """

    :param ray_source:
    :param ray_direction:
    :param sphere_list:
    :param depth:
    :return:
    """
    bias = 1e-4

    sphere, pt_tuple = min(((s, s.intersection(ray_source, ray_direction)) for s in sphere_list),
                           key=lambda tu: tu[1][1])
    hit_point, _ = pt_tuple
    if hit_point is None:
        return np.array([2., 2., 2.], dtype=np.float64)

    normal_vec = utils.normalize(hit_point - sphere.center)
    is_inside = False
    if np.dot(ray_direction, normal_vec) > 0:
        # ray_source 在 sphere 内
        normal_vec = -normal_vec
        is_inside = True

    surface_color = np.array([0., 0., 0.], dtype=np.float64)
    if (sphere.transparency > 0 or sphere.reflection > 0) and depth < MAX_DEPTH:
        cos_theta = - np.asscalar(np.dot(normal_vec, ray_direction))

        # 计算反射光线的方向
        reflection_direction = ray_direction - 2 * np.dot(ray_direction, normal_vec) * normal_vec
        reflection_direction = utils.normalize(reflection_direction)

        # 稍微偏移, 不让起点在球面上
        reflection = ray_trace(hit_point + bias * normal_vec, reflection_direction, sphere_list, depth + 1)

        refraction = np.array([0., 0., 0.], dtype=np.float64)
        if sphere.transparency > 0:
            eta = sphere.refractive_rate
            if not is_inside:
                eta = 1 / eta
            try:
                cos_alpha = math.sqrt(1 - (1 - cos_theta * cos_theta) * eta * eta)
            except ValueError:
                pass
            else:
                # 计算折射光线方向
                refraction_direction = eta * ray_direction + (eta * cos_theta - cos_alpha) * normal_vec
                refraction_direction = utils.normalize(refraction_direction)
                refraction = ray_trace(hit_point - bias * normal_vec,
                                       refraction_direction,
                                       sphere_list,
                                       depth=depth + 1)

        fresnel_coeff = mix((1 - cos_theta) ** 3, 1, 0.1)
        surface_color = (reflection * fresnel_coeff
                         + refraction * (1 - fresnel_coeff) * sphere.transparency) * sphere.surface_color

    else:
        for i, sphere_i in enumerate(sphere_list):
            if any(sphere_i.emission_color > 0):
                transmission = 1
                light_direction = sphere_i.center - hit_point
                light_direction = utils.normalize(light_direction)
                for j, sphere_j in enumerate(sphere_list):
                    if i == j:
                        continue
                    pt, _ = sphere_j.intersection(hit_point + bias * normal_vec, light_direction)
                    if pt is not None:
                        # 光线被阻挡
                        transmission = 0
                        break
                surface_color += (sphere.surface_color
                                  * transmission
                                  * max(0., np.dot(normal_vec, light_direction))
                                  * sphere_i.emission_color)
    return surface_color + sphere.emission_color


def _pixel_iter(width: int,
                height: int,
                inv_width: float,
                inv_height: float,
                tan_theta: float,
                ratio: float):
    """

    :param width:
    :param height:
    :param inv_width:
    :param inv_height:
    :param tan_theta:
    :param ratio:
    :return:
    """
    for i in range(height):
        for j in range(width):
            x = (2 * (j + 0.5) * inv_width - 1) * tan_theta * ratio
            y = (1 - (2 * (i + 0.5) * inv_height)) * tan_theta
            r_direction = np.array([x, y, -1])
            r_direction = utils.normalize(r_direction)
            yield r_direction, i, j


def _modify_pixel(pixel):
    """

    :param pixel:
    :return:
    """
    return np.array([int(max(min(x, 1), 0) * 255.99) for x in pixel], dtype=np.int32)


def render(sphere_list: List[geometric.Sphere],
           width: int = 640,
           height: int = 480,
           view_angle: float = 30,
           source_point: Union[None, np.ndarray] = None,
           use_multiprocess: bool = False):
    """

    :param sphere_list:
    :param width:
    :param height:
    :param view_angle:
    :param source_point:
    :param use_multiprocess:
    :return:
    """
    if source_point is None:
        source_point = np.array([0., 0., 0.], dtype=np.float)
    image = np.zeros(shape=(width, height, 3), dtype=np.int32)

    inv_width = 1 / width
    inv_height = 1 / height
    tan_theta = math.tan(math.radians(view_angle) / 2)
    ratio = width / height

    light_iter = _pixel_iter(width, height, inv_width, inv_height, tan_theta, ratio)
    if use_multiprocess:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            future_dict = {executor.submit(ray_trace, source_point, input_tuple[0], sphere_list, 0): input_tuple[1:]
                           for input_tuple in light_iter}
            for future in concurrent.futures.as_completed(future_dict):
                i, j = future_dict[future]
                try:
                    pixel = future.result()
                except Exception as e:
                    logging.error(e)
                    pixel = np.array([0, 0, 0], dtype=np.int32)
                image[j][i] = _modify_pixel(pixel)
    else:
        for input_tuple in light_iter:
            r_direction, i, j = input_tuple
            try:
                pixel = ray_trace(source_point, r_direction, sphere_list, 0)
            except Exception as e:
                logging.error(e, exc_info=True)
                pixel = np.array([0, 0, 0], dtype=np.int32)
            image[j][i] = _modify_pixel(pixel)
    return image


if __name__ == '__main__':

    sphere_list = []
    sphere_list.append(geometric.Sphere(np.array([0.0, -10004, -20]),
                                        10000,
                                        1.1,
                                        np.array([0.20, 0.20, 0.20]),
                                        None, 0, 0.0))
    sphere_list.append(geometric.Sphere(np.array([0.0, 0, -20]), 4, 1.1, np.array([1.00, 0.32, 0.36]), None, 1, 0.5))
    sphere_list.append(geometric.Sphere(np.array([5.0, -1, -15]), 2, 1.1, np.array([0.90, 0.76, 0.46]), None, 1, 0.0))
    sphere_list.append(geometric.Sphere(np.array([5.0, 0, -25]), 3, 1.1, np.array([0.65, 0.77, 0.97]), None, 1, 0.0))
    sphere_list.append(geometric.Sphere(np.array([-5.5, 0, -15]), 3, 1.1, np.array([0.90, 0.90, 0.90]), None, 1, 0.0))
    now = time.time()
    image = render(sphere_list)
    run_time = time.time() - now
    filename = 'test.ppm'
    print(image.shape)
    utils.write_ppm(image, filename)
    print(f'时间: {run_time}')