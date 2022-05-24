from typing import Union

import numpy as np


def _to_array(a: Union[int, np.ndarray]) -> np.ndarray:
    if type(a) is int:
        a = np.ones(2, dtype=int) * a
    return a


def conv_2d(
    input_size: Union[int, np.ndarray],
    kernel_size: Union[int, np.ndarray],
    stride: Union[int, np.ndarray],
    padding: Union[int, np.ndarray] = 0,
) -> np.ndarray:
    """
    Returns the size of the output of a Conv2D

    :param input_size: The input size
    :type input_size: Union[int, np.ndarray]
    :param kernel_size: The kernel size
    :type kernel_size: Union[int, np.ndarray]
    :param stride: The stride
    :type stride: Union[int, np.ndarray]
    :param padding: The padding defaults to 0
    :type padding: Union[int, np.ndarray], optional
    :return: The size of the output as numpy array
    :rtype: np.ndarray
    """
    input_size = _to_array(input_size)
    kernel_size = _to_array(kernel_size)
    stride = _to_array(stride)
    padding = _to_array(padding)
    assert ((input_size - kernel_size + 2 * padding) % stride == 0).all()
    return (input_size - kernel_size + 2 * padding) // stride + 1


def conv_transpose_2d(
    input_size: Union[int, np.ndarray],
    kernel_size: Union[int, np.ndarray],
    stride: Union[int, np.ndarray],
    padding: Union[int, np.ndarray] = 0,
    output_padding: Union[int, np.ndarray] = 0,
) -> np.ndarray:
    """_summary_

    :param input_size: The input size
    :type input_size: Union[int, np.ndarray]
    :param kernel_size: The kernel size
    :type kernel_size: Union[int, np.ndarray]
    :param stride: The stride
    :type stride: Union[int, np.ndarray]
    :param padding: The padding defaults to 0
    :type padding: Union[int, np.ndarray], optional
    :param output_padding: The output padding, defaults to 0
    :type output_padding: Union[int, np.ndarray], optional
    :return: The size of the output as numpy array
    :rtype: np.ndarray
    """
    input_size = _to_array(input_size)
    kernel_size = _to_array(kernel_size)
    stride = _to_array(stride)
    padding = _to_array(padding)
    return (input_size - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1


def pool_2d(
    input_size: Union[int, np.ndarray],
    kernel_size: Union[int, np.ndarray],
    stride: Union[int, np.ndarray],
    padding: Union[int, np.ndarray] = 0,
) -> np.ndarray:
    """
    Returns the size of the output of a Pool2D

    :param input_size: The input size
    :type input_size: Union[int, np.ndarray]
    :param kernel_size: The kernel size
    :type kernel_size: Union[int, np.ndarray]
    :param stride: The stride
    :type stride: Union[int, np.ndarray]
    :param padding: The padding defaults to 0
    :type padding: Union[int, np.ndarray], optional
    :return: The size of the output as numpy array
    :rtype: np.ndarray
    """
    return conv_2d(input_size, kernel_size, stride, padding)
