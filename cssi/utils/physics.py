import math
import numpy as np


def calculate_euler_angles(rmat, inverse=False):
    """Calculates the Euler angles when a rotation matrix is passed in"""
    assert (_is_rotation_matrix(rmat))

    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

    singular = sy < 1e-6

    if not singular:
        if inverse:
            x = math.atan2(-rmat[2, 1], rmat[2, 2])
            y = math.atan2(rmat[2, 0], sy)
            z = math.atan2(-rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        if inverse:
            x = math.atan2(rmat[1, 2], rmat[1, 1])
            y = math.atan2(rmat[2, 0], sy)
            z = 0
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0

    x = x * 180.0 / math.pi
    y = y * 180.0 / math.pi
    z = z * 180.0 / math.pi

    return np.array([x, y, z])


def _is_rotation_matrix(R):
    """Checks if a matrix is a valid rotation matrix."""
    rt = np.transpose(R)
    should_be_identity = np.dot(rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def calculate_angle_diff(angle_1, angle_2):
    return 180 - (180 - angle_2 + angle_1) % 360
