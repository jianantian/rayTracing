import math

import numpy as np

SMALL_EPSILON = 1e-6


def square_norm(vec: np.ndarray) -> float:
    """

    :param vec:
    :return:
    """
    return sum(x ** 2 for x in vec)


def norm(vec: np.ndarray) -> float:
    """

    :param vec:
    :return:
    """
    return math.sqrt(square_norm(vec))


def normalize(array: np.ndarray, eps: float = SMALL_EPSILON):
    """
    array 归一化
    :param array: np.array
    :param eps:
    :return: np.array
    """
    if norm(array) < eps:
        raise ValueError('array 长度太短: {}'.format(array))
    res = array / norm(array)
    return res


def write_ppm(img: np.ndarray, filename: str) -> None:
    """
    以 ppm3 图片格式保存 np.ndarray
    """
    width, height, chanel = img.shape
    with open(filename, 'w') as fr:
        fr.write(f'P3 \n{width} {height} \n255\n')
        for j in range(height):
            for i in range(width):
                pixel = img[i][j]
                for x in pixel:
                    fr.write(f'{x} ')
                fr.write('\n')


def write_ppm_np(img: np.ndarray, filename: str) -> None:
    """
    以 ppm3 图片格式保存 np.ndarray
    """
    width, height, chanel = img.shape
    with open(filename, 'w') as fr:
        fr.write(f'P3 \n{width} {height} \n255')
        for i, x in enumerate(np.nditer(img.swapaxes(0, 1), order='C')):
            if i % 3 == 0:
                fr.write('\n')
            fr.write(f'{x} ')
