import PyKDL as kdl
import numpy as np

# def joint_list_to_kdl(q):
#     if q is None:
#         return None
#     if type(q) == np.matrix and q.shape[1] == 0:
#         q = q.T.tolist()[0]
#
#     q_kdl = kdl.JntArray(len(q))
#     for i, q_i in enumerate(q):
#         q_kdl[i] = q_i
#
#     return q_kdl


def joint_to_kdl_jnt_array(q):
    if isinstance(q, np.ndarray) and q.ndim == 1:
        q_kdl = kdl.JntArray(q.size)
        for i in range(q.size):
            q_kdl[i] = q[i]

    elif isinstance(q, list):
        q_kdl = kdl.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i

    else:
        raise ValueError("Joint Vector q must be either a np.ndarray or list but is type {0}.".format(type(q)))

    return q_kdl


def kdl_jnt_array_to_joint(vec_kdl):
    assert isinstance(vec_kdl, kdl.JntArray)
    vec = np.zeros(vec_kdl.rows())

    for i in range(vec_kdl.rows()):
        vec[i] = vec_kdl[i]

    return vec


def kdl_vector_to_vector(vec_kdl):
    assert isinstance(vec_kdl, kdl.Vector), "Vector must be type 'kdl.Vector' but is '{0}'".format(type(vec_kdl))
    return np.array([vec_kdl.x(), vec_kdl.y(), vec_kdl.z()])


def vector_to_kdl_vector(x):
    assert (isinstance(x, np.ndarray) and x.size == 3) or (isinstance(x, list) and len(x) == 3)
    return kdl.Vector(x[0], x[1], x[2])


def kdl_inertia_to_matrix(inertia_kdl):
    inertia = np.zeros((3, 3))
    ix, iy = np.unravel_index(np.arange(9), (3, 3))
    for i in range(9):
        inertia[ix[i], iy[i]] = inertia_kdl[i]

    return inertia


def rot_x(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct Matrix:
    mat = np.eye(3)
    mat[1, 1] = + cos_theta
    mat[1, 2] = - sin_theta
    mat[2, 1] = + sin_theta
    mat[2, 2] = + cos_theta
    return mat


def rot_y(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct Matrix:
    mat = np.eye(3)
    mat[0, 0] = + cos_theta
    mat[0, 2] = + sin_theta
    mat[2, 0] = - sin_theta
    mat[2, 2] = + cos_theta
    return mat


def rot_z(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct Matrix:
    mat = np.eye(3)
    mat[0, 0] = + cos_theta
    mat[0, 1] = - sin_theta
    mat[1, 0] = + sin_theta
    mat[1, 1] = + cos_theta
    return mat


def kdl_matrix_to_mat(mat_kdl):
    mat = np.zeros((mat_kdl.rows(), mat_kdl.columns()))

    for i in range(mat_kdl.rows()):
        for j in range(mat_kdl.columns()):
            mat[i, j] = mat_kdl[i, j]

    return mat


def kdl_rot_to_mat(rot):
    return np.array([[rot[0, 0], rot[0, 1], rot[0, 2]],
                     [rot[1, 0], rot[1, 1], rot[1, 2]],
                     [rot[2, 0], rot[2, 1], rot[2, 2]]])


def kdl_frame_to_hom_transformation_matrix(frame):
    p = frame.p
    m = frame.M
    return np.array([[m[0, 0], m[0, 1], m[0, 2], p.x()],
                     [m[1, 0], m[1, 1], m[1, 2], p.y()],
                     [m[2, 0], m[2, 1], m[2, 2], p.z()],
                     [0, 0, 0, 1]])


def kdl_frame_to_transformation_matrix(frame):
    p = frame.p
    m = frame.M
    return np.array([[m[0, 0], m[0, 1], m[0, 2], p.x()],
                     [m[1, 0], m[1, 1], m[1, 2], p.y()],
                     [m[2, 0], m[2, 1], m[2, 2], p.z()]])


def rotation_mat_distance(mat_rot_1, mat_rot_2):
    mat_r = np.dot(mat_rot_1, mat_rot_2.transpose())
    theta = np.arccos((np.trace(mat_r) - 1.) / 2.)

    return theta

