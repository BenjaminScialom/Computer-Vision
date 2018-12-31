import argparse
import sys
import numpy as np

########################################################################################################################
########################################################################################################################

# random_number n -> random.randint(1,len(d3_pts)-6)
# -> for list_3D, list_2D in zip(d3pts[n:n+6, d2pts[n:n+6]) -> issue => take all points

def construct_matrix_A(d3_pts, d2_pts):

    # Matrix initialization
    A = []
    # Vector of zeros used to construct the matrix form of the 2 equations
    zeros = np.zeros(4)

    for i, j in zip(d3_pts, d2_pts):

        # Using homogeneous coordinate for each point
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        # Utilization of the two equation established in the course.
        # These equation are based on projection equation and equation from shift in homogeneous coordinates
        xi_pi = j[0] * pi
        yi_pi = j[1] * pi
        # Construction of the matrix
        A.append(np.concatenate([pi, zeros, -xi_pi]))
        A.append(np.concatenate([zeros, pi, -yi_pi]))

    return np.array(A)


########################################################################################################################
#######################################################################################################################


def construct_Matrix_M(A):

    # Use SVD to get M in course
    U, D, V = np.linalg.svd(A, full_matrices=True)

    # X which contains the vectors m1,m2,m3 is the vector V belonging to the zero singular value
    x = V[-1].reshape(3, 4)

    return x
########################################################################################################################
########################################################################################################################


def compute_mean_square_error(M, d3_pts, d2_pts):

    mse = 0
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
        mse += np.sqrt(((xi - (m_2.T.dot(point_3DH)) / (m_3.T.dot(point_3DH))) ** 2 + (yi - (m_1.T.dot(point_3DH)) /
                                                                                       (m_3.T.dot(point_3DH))) ** 2))

    return mse

########################################################################################################################
########################################################################################################################


def read_points_file(file):
    obj_pts, img_pts = [], []

    coord_3D = obj_pts.append
    coord_2D = img_pts.append
    for line in file.readlines():
        # split data into a vector
        vect = line.split()

        # The 3 first are 3D coordinate and the 2 next are 2D coordinate
        coord_3D(list(map(float, vect[:3])))
        coord_2D(list(map(float, vect[3:])))

    # Close file at the end of utilization
    file.close()

    return obj_pts, img_pts

########################################################################################################################
########################################################################################################################


def calibratation_perso(d3pts, d2pts):

    # Matrix A constructed thanks to 6 points in the data file : 2 equations per point
    A = construct_matrix_A(d3pts, d2pts)
    # print(a) -> no error here

    # Matrix M constructed using SVD and taking the vector belonging to the 0 singular value
    M = construct_Matrix_M(A)

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
    print("R* is : ")
    print(R_star)
    print("\n")

    # Computation of the mean square :
    mse = compute_mean_square_error(M, d3pts, d2pts)
    print("Means square error: \n", mse)

    return u0, v0, alpha_u, alpha_v, S, T_star, R_star, mse

########################################################################################################################
########################################################################################################################


def main():

    # Select the file in argument as the one to use
    parser_description = "You are using  the basic calibration"
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument('--data', '-d', help="Give the path of the 2D-3D correspondence file in 1st argument",
                        default=None)
    args = parser.parse_args()
    data_file = args.data

    # It should not work if nothing is add
    if data_file == None:
        parser.print_help()
        sys.exit()

    # Open data and select 2D point and 3D points separately
    file = open(data_file, 'r')
    objpoints, imgpoints = read_points_file(file)
    print("End of data reading !")

    # Process the basic calibration
    print("\n Starting Non-planar Calibration ...")
    calibratation_perso(objpoints, imgpoints)
    print("Calibration using own non planar function : done ")

########################################################################################################################


if __name__ == '__main__':
        main()


########################################################################################################################
########################################################################################################################
