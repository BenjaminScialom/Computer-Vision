import numpy as np
import math
import configparser
import argparse
import random
import sys

########################################################################################################################
########################################################################################################################


def construct_matrix_A(d3_pts, d2_pts):

    A = []
    zeros = np.zeros(4)
    # random_number n -> random.randint(1,len(d3_pts)-6)
    # -> for list_3D, list_2D in zip(d3pts[n:n+6, d2pts[n:n+6]) -> issue => take all points

    for list_3D, list_2D in zip(d3_pts, d2_pts):

        # Using homogeneous coordinate for each point
        point_i = np.array(list_3D)
        point_i = np.concatenate([point_i, [1]])
        # Utilization of the two equation established in the course.
        # These equation are based on projection equation and equation from shift in homogeneous coordinates
        xi = list_2D[0]*point_i
        yi = list_2D[1]*point_i
        # Construction of the matrix
        A.append(np.concatenate([point_i, zeros, -xi]))
        A.append(np.concatenate([zeros, point_i, -yi]))

    return np.array(A)

########################################################################################################################
#######################################################################################################################


def construct_Matrix_M(A):

    # Use SVD to get M in course
    U, D, V = np.linalg.svd(A, full_matrices=True)

    # X which contains the vectors m1,m2,m3 is the vector V belonging to the zero singular value
    M = V[-1].reshape(3, 4)

    return M
########################################################################################################################
########################################################################################################################


def compute_mean_square_error(M, d3_pts, d2_pts):

    mse = []
    m_1 = M[0]
    m_2 = M[1]
    m_3 = M[2]

    for i, j in zip(d3_pts, d2_pts):

        # Get 2D coordinate x and y which are the references
        xi = j[0]
        yi = j[1]

        # Make 3DH points instead of 3D in order that size match
        point_3DH = np.array(i)
        point_3DH = np.append(point_3DH, 1)

        # Compute the mean square error according to the course formula
        mse.append(np.sqrt(((xi - (m_2.T.dot(point_3DH)) / (m_3.T.dot(point_3DH))) ** 2 + (yi - (m_1.T.dot(point_3DH)) /
                                                                                       (m_3.T.dot(point_3DH))) ** 2)))

    return mse

########################################################################################################################
########################################################################################################################


def read_points_file(file):
    obj_pts, img_pts = [], []

    coord_3D = obj_pts.append
    coord_2D = img_pts.append
    with open(file) as f:
        for line in f.readlines():

            # split data into a vector
            vect = line.split()

            # The 3 first are 3D coordinate and the 2 next are 2D coordinate
            coord_3D(list(map(float, vect[:3])))
            coord_2D(list(map(float, vect[3:])))

    return obj_pts, img_pts

########################################################################################################################
########################################################################################################################


def calibratation_perso(M):

    # Get the 3 component of the vector M
    # m1 transpose
    m1 = M[0][:3].T
    # m2 transpose
    m2 = M[1][:3].T
    # m3 transpose
    m3 = M[2][:3].T

    # Vector b which correspond at the last column of the matrix M
    b = np.reshape(M[:, 3], (3, 1))

    # Compute intrinsic parameters
    norm_rho = 1 / np.linalg.norm(m3)
    u0 = norm_rho ** 2 * (m1.dot(m3.T))
    v0 = norm_rho ** 2 * (m2.dot(m3.T))
    alpha_v = np.sqrt(norm_rho ** 2 * m2.dot(m2.T) - v0 ** 2)
    S = (norm_rho ** 4) / alpha_v * np.cross(m1, m3).dot(np.cross(m2, m3).T)
    alpha_u = np.sqrt(norm_rho ** 2 * m1.dot(m1.T) - S ** 2 - u0 ** 2)
    K_star = np.array([[alpha_u, S, u0], [0, alpha_v, v0], [0, 0, 1]])

    # Display intrinsic parameters:
    print("\n")
    print("Alpha_u :")
    print(alpha_u)
    print("\n")
    print("Alpha_v :")
    print(alpha_v)
    print("\n")
    print("U0 :")
    print(u0)
    print("\n")
    print("v0 :")
    print(v0)
    print("\n")
    print("K* :")
    print(K_star)
    print("\n")
    print("S :")
    print(S)
    print("\n")

    # Compute extrinsic parameters
    sign = np.sign(b[2])
    T_star = sign * norm_rho * np.linalg.inv(K_star).dot(b).T
    r1 = norm_rho ** 2 / alpha_v * np.cross(m2, m3)
    r3 = sign * norm_rho * m3.T
    r2 = np.cross(r3, r1)
    R_star = np.array([r1.T, r2.T, r3.T])

    # Display extrinsic parameters
    print("T* is :")
    print(T_star)
    print("\n")
    print("R* is :")
    print(R_star)

    return u0, v0, alpha_u, alpha_v, K_star, S, T_star, R_star

########################################################################################################################
########################################################################################################################


def ransac_method(objt_pts, img_pts, prob, k_max, max_samples_number, min_samples_number,):

    # Initialization of the Ransac parameters
    counter = 0
    ransac_M = None
    inliner_number = 0
    # k is the number of trials
    k = k_max
    # initialization of matrix M thanks to A based on the points from the data file
    A_initial = construct_matrix_A(objt_pts, img_pts)
    M_initial = construct_Matrix_M(A_initial)
    # Initialization of the mean square error
    initial_mse = compute_mean_square_error(M_initial, objt_pts, img_pts)
    # inliner threshold
    t = 3*np.median(initial_mse)
    # The number of points is limited by the value put in the Ransac configuration file
    n = random.randint(min_samples_number, max_samples_number)

    # Perform multiple experiment k times until
    while counter < k and counter < k_max:

        index = np.random.choice(len(objt_pts), n)
        ransac_objt_pts, ransac_img_pts = np.array(objt_pts)[index], np.array(img_pts)[index]
        A = construct_matrix_A(ransac_objt_pts, ransac_img_pts)
        M = construct_Matrix_M(A)
        mse = compute_mean_square_error(M, objt_pts, img_pts)
        mse_list = list(mse)
        list_inliner = []

        for i, d in enumerate(mse_list):
            # Find inliners in the entire set using the condition d<t
            if d < t:
                list_inliner.append(i)

        # update of w for each iteration
        w = float(len(list_inliner))/float(len(img_pts))

        # Choose the best result for among all that we have if we have enough inliner
        if len(list_inliner) >= inliner_number:
            inliner_number = len(list_inliner)
            A = construct_matrix_A(ransac_objt_pts, ransac_img_pts)
            ransac_M = construct_Matrix_M(A)

        if not (w == 0):
            # Equation build from the assumption that all k experiment failed
            # (1-p)=(1-w**n)**k -> apply log and isolate k
            # Update of k for each iteration
            k = float(math.log(1 - prob)) / np.absolute(math.log(1 - (w ** n)))

        counter += 1

    return inliner_number, ransac_M

########################################################################################################################
########################################################################################################################


def main():

    parser_description = "You are using a calibration method improved with Ransac Algorithm"

    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument("--configuration", '-c',  help="You shouldGive the path of the configuration file\
     you want to use", default='RANSAC.config')

    parser.add_argument('--data', '-d', help="Give the path of the 2D-3D correspondence file in 1st argument",
                        default=None)

    args = parser.parse_args()
    file_name = args.data
    config_path = args.configuration

    # It should not work if nothing is add
    if file_name == None:
        parser.print_help()
        sys.exit()

    # Get the value
    config = configparser.ConfigParser()
    config.read(config_path)
    p = config['CONF']['p']
    min_samples_number = config['CONF']['min_samples_number']
    max_samples_number = config['CONF']['max_samples_number']
    k_max = config['CONF']['k_max']

    # Convert to float
    p = float(p)
    min_samples_number = float(min_samples_number)
    max_samples_number = float(max_samples_number)
    k_max = float(k_max)

    # Process calibration using Ransac
    objt_points, img_points = read_points_file(file_name)
    inliner_num, ransac_M = ransac_method(objt_points, img_points, p, k_max, max_samples_number, min_samples_number)
    calibratation_perso(ransac_M)


########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    main()


########################################################################################################################
########################################################################################################################
