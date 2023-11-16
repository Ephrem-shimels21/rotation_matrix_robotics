import numpy as np
import math


def rotation_matrix_to_xyz_fixed_angles(rot_matrix):
    rot_matrix = np.asarray(rot_matrix)
    r11, r12, r13 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    r21, r22, r23 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    r31, r32, r33 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]

    beta = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
    alpha = np.arctan2(r21 / np.cos(beta), r11 / np.cos(beta))
    gamma = np.arctan2(r32 / np.cos(beta), r33 / np.cos(beta))

    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)


def rotation_matrix_to_zyz_euler_angles(rot_matrix):
    rot_matrix = np.asarray(rot_matrix)
    r11, r12, r13 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    r21, r22, r23 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    r31, r32, r33 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]

    beta = np.arccos(r33)
    alpha = np.arctan2(r23 / np.sin(beta), r13 / np.sin(beta))
    gamma = np.arctan2(r32 / np.sin(beta), -r31 / np.sin(beta))

    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)


def rotation_matrix_to_euler_parameters(rot_matrix):
    rot_matrix = np.asarray(rot_matrix)
    r11, r12, r13 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    r21, r22, r23 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    r31, r32, r33 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]

    q0 = 0.5 * np.sqrt(1 + r11 + r22 + r33)
    q1 = (r32 - r23) / (4 * q0)
    q2 = (r13 - r31) / (4 * q0)
    q3 = (r21 - r12) / (4 * q0)

    return q0, q1, q2, q3


def xyz_fixed_angles_to_rotation_matrix(alpha, beta, gamma):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

    R_y = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    R_z = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    return np.dot(np.dot(R_z, R_y), R_x)


def zyz_euler_angles_to_rotation_matrix(alpha, beta, gamma):
    R_z1 = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )

    R_y = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    R_z2 = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    return np.dot(np.dot(R_z2, R_y), R_z1)


def euler_parameters_to_rotation_matrix(q0, q1, q2, q3):
    R = np.array(
        [
            [
                1 - 2 * (q2**2 + q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q0 * q2 + q1 * q3),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1**2 + q3**2),
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q0 * q1 + q2 * q3),
                1 - 2 * (q1**2 + q2**2),
            ],
        ]
    )

    return R


# Example usage:
rot_matrix = np.array(
    [[0.6124, 0.6124, -0.5], [-0.3536, 0.6124, 0.7071], [0.7071, -0.5, 0.5]]
)
# rot_matrix = np.array(
#     [[0.6124, -0.3536, 0.7071], [0.6124, 0.6124, -0.5], [-0.5, 0.7071, 0.5]]
# )
# Convert rotation matrix to X-Y-Z fixed angles
print(rot_matrix)
xyz_fixed_angles = rotation_matrix_to_xyz_fixed_angles(rot_matrix)
print("X-Y-Z Fixed Angles:", xyz_fixed_angles)

# Convert rotation matrix to Z-Y-Z Euler angles
zyz_euler_angles = rotation_matrix_to_zyz_euler_angles(rot_matrix)
print("Z-Y-Z Euler Angles:", zyz_euler_angles)

# Convert rotation matrix to Euler parameters (quaternions)
euler_parameters = rotation_matrix_to_euler_parameters(rot_matrix)
print("Euler Parameters (Quaternions):", euler_parameters)

# Convert X-Y-Z fixed angles to rotation matrix
rot_matrix_xyz = xyz_fixed_angles_to_rotation_matrix(*xyz_fixed_angles)
print("Rotation Matrix from X-Y-Z Fixed Angles:")
print(rot_matrix_xyz)

# Convert Z-Y-Z Euler angles to rotation matrix
rot_matrix_zyz = zyz_euler_angles_to_rotation_matrix(*zyz_euler_angles)
print("Rotation Matrix from Z-Y-Z Euler Angles:")
print(rot_matrix_zyz)

# Convert Euler parameters to rotation matrix
rot_matrix_quaternions = euler_parameters_to_rotation_matrix(*euler_parameters)
print("Rotation Matrix from Euler Parameters:")
print(rot_matrix_quaternions)


# def fixed_angles_to_rot_matrix(Beta, Alpha, Gamma):
#     Beta = np.radians(Beta)
#     Alpha = np.radians(Alpha)
#     Gamma = np.radians(Gamma)

#     rot_mat_x = [
#                  np.cos(Alpha) * np.cos(Beta),
#                  np.sin(Alpha) * np.cos(Beta),
#                  -np.sin(Beta)
#                  ]

#     rot_mat_y = [
#                  np.cos(Alpha) * np.sin(Beta) * np.sin(Gamma) - np.sin(Alpha) * np.cos(Gamma),
#                  np.sin(Alpha) * np.sin(Beta) * np.sin(Gamma) + np.cos(Alpha) * np.cos(Gamma),
#                  np.cos(Beta) * np.sin(Gamma)
#                  ]
#     rot_mat_z = [
#           np.cos(Alpha) * np.sin(Beta) * np.cos(Gamma) + np.sin(Alpha) * np.sin(Gamma),
#           np.sin(Alpha) * np.sin(Beta) * np.cos(Gamma) - np.cos(Alpha) * np.sin(Gamma),
#           np.cos(Beta) * np.cos(Gamma)
#           ]

#     rot_matrix = np.stack([rot_mat_x, rot_mat_y, rot_mat_z])

#     return rot_matrix
