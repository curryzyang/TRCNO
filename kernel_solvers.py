"""Kernel solver utilities extracted from the notebook.
Provides solver_2x2 and K_solver_2x2 functions for reuse.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapz


def solver_2x2(params, N_g):
    """Numerical solver for the 2x2 kernel system.

    Inputs:
      params: object with arrays mu, lam, a_1, a_2, b_1, b_2, f, q
      N_g: number of grid points
    Outputs:
      K, L: (N_g x N_g) kernel matrices (numpy arrays)
    """
    Delta = 1 / (N_g - 1)
    numTot = int(N_g * (N_g + 1) / 2)  # Total number of nodes per kernel

    index_map = np.zeros((N_g, N_g))
    index_map[np.tril_indices(N_g)] = np.arange(1, numTot + 1)
    index_map = np.transpose(index_map)

    dummy1 = np.tril(np.reshape(np.arange(1, (N_g ** 2) + 1), (N_g, N_g))).transpose()
    glob_to_mat = dummy1[dummy1 > 0]
    glob_to_mat = np.array(sorted(glob_to_mat))

    indexes = np.kron(np.arange(1, (2 * numTot) + 1), np.ones(5)).reshape((2 * numTot, 5)).astype(int)
    indexes = indexes - 1
    weights = np.zeros((2 * numTot, 5))
    RHS = np.zeros((2 * numTot, 1))

    # Directional derivatives
    s_K_x, s_K_y = np.meshgrid(params.mu, -params.lam)
    s_L_x, s_L_y = np.meshgrid(params.mu, params.mu)

    s_K = np.sqrt(s_K_x ** 2 + s_K_y ** 2)
    s_L = np.sqrt(s_L_x ** 2 + s_L_y ** 2)

    # K: Iterate over triangular domain
    for indX in range(N_g):
        for indY in range(indX + 1):
            glob_ind = int(index_map[indY, indX])
            if (indY == indX):  # Boundary
                weights[glob_ind - 1, 0] = 1
                RHS[glob_ind - 1] = params.f[indY]
            else:
                theta = np.arctan2(s_K_x[indY, indX], -s_K_y[indY, indX])
                if theta > np.pi / 4:
                    # Then it hits the left side
                    sigma = Delta / np.sin(theta)
                    # Vector length
                    d = sigma * np.cos(theta)
                    # Distance from top
                    preFactor = (s_K[indY, indX] / sigma)
                    # Itself
                    weights[glob_ind - 1, 0] = preFactor
                    # Top left
                    weights[glob_ind - 1, 1] = -preFactor * d / Delta
                    indexes[glob_ind - 1, 1] = int(index_map[indY + 1, indX - 1]) - 1

                    # Bottom left
                    weights[glob_ind - 1, 2] = -preFactor * (Delta - d) / Delta
                    indexes[glob_ind - 1, 2] = int(index_map[indY, indX - 1]) - 1
                else:
                    # Then it hits the top
                    sigma = Delta / np.cos(theta)
                    # Vector length
                    d = sigma * np.sin(theta)
                    # Distance from top
                    preFactor = (s_K[indY, indX] / sigma)
                    # Itself
                    weights[glob_ind - 1, 0] = preFactor
                    # Top left
                    weights[glob_ind - 1, 1] = -preFactor * d / Delta
                    indexes[glob_ind - 1, 1] = int(index_map[indY + 1, indX - 1]) - 1
                    # Top right
                    weights[glob_ind - 1, 2] = -preFactor * (Delta - d) / Delta
                    indexes[glob_ind - 1, 2] = int(index_map[indY + 1, indX]) - 1

                if indY == indX - 1:
                    # Subdiagonal: top-left node does not exist; redistribute
                    if theta > np.pi / 4:  # Then it hits the left side
                        weights[glob_ind - 1, 2] = weights[glob_ind - 1, 2] + weights[glob_ind - 1, 1]
                        weights[glob_ind - 1, 0] = weights[glob_ind - 1, 0] - weights[glob_ind - 1, 1]
                        indexes[glob_ind - 1, 1] = int(index_map[indY + 1, indX]) - 1
                    else:  # Then it hits the top side
                        weights[glob_ind - 1, 2] = weights[glob_ind - 1, 2] + weights[glob_ind - 1, 1]
                        weights[glob_ind - 1, 0] = weights[glob_ind - 1, 0] - weights[glob_ind - 1, 1]
                        indexes[glob_ind - 1, 1] = int(index_map[indY, indX - 1]) - 1

    for indX in range(N_g):
        for indY in range(indX + 1):
            # L: Iterate over triangular domain
            glob_ind = int(index_map[indY, indX]) + numTot
            if indY == 0:
                # Boundary
                weights[glob_ind - 1, 0] = -1
                weights[glob_ind - 1, 1] = params.q
                indexes[glob_ind - 1, 1] = glob_ind - numTot - 1
            else:
                theta = np.arctan2(s_L_y[indY, indX], s_L_x[indY, indX])
                if theta < np.pi / 4:
                    # Then it hits the left side
                    sigma = Delta / np.cos(theta)
                    # Vector length
                    d = sigma * np.sin(theta)
                    # Distance from top
                    preFactor = (s_L[indY, indX] / sigma)
                    # Itself
                    weights[glob_ind - 1, 0] = preFactor
                    # Top left
                    weights[glob_ind - 1, 1] = -preFactor * (Delta - d) / Delta
                    indexes[glob_ind - 1, 1] = int(index_map[indY, indX - 1]) + numTot - 1

                    # Bottom left
                    weights[glob_ind - 1, 2] = -preFactor * d / Delta
                    indexes[glob_ind - 1, 2] = int(index_map[indY - 1, indX - 1]) + numTot - 1
                else:
                    # Then it hits the bottom
                    sigma = Delta / np.sin(theta)
                    # Vector length
                    d = sigma * np.cos(theta)
                    # Distance from right
                    preFactor = (s_L[indY, indX] / sigma)
                    # Itself
                    weights[glob_ind - 1, 0] = preFactor

                    # Bottom left
                    weights[glob_ind - 1, 1] = -preFactor * d / Delta
                    indexes[glob_ind - 1, 1] = int(index_map[indY - 1, indX - 1]) + numTot - 1

                    # Bottom right
                    weights[glob_ind - 1, 2] = -preFactor * (Delta - d) / Delta
                    indexes[glob_ind - 1, 2] = int(index_map[indY - 1, indX]) + numTot - 1

            # Source terms
    for indX in range(N_g):
        for indY in range(indX + 1):
            glob_ind = int(index_map[indY, indX])
            if indY != indX:
                weights[glob_ind - 1, 3] = -params.a_1[indY, indX]
                weights[glob_ind - 1, 4] = -params.a_2[indY, indX]
                indexes[glob_ind - 1, 4] = glob_ind + numTot - 1

            if indY != 0:
                weights[glob_ind - 1 + numTot, 3] = -params.b_1[indY, indX]
                indexes[glob_ind - 1 + numTot, 3] = glob_ind - 1
                weights[glob_ind - 1 + numTot, 4] = -params.b_2[indY, indX]

    ## Matrix solving
    counter = 0
    A_coord_x = np.zeros(2 * numTot * 5, dtype=int)
    A_coord_y = np.zeros(2 * numTot * 5, dtype=int)
    A_vals = np.zeros(2 * numTot * 5)

    b_coord_y = np.zeros(2 * numTot, dtype=int)
    b_vals = np.zeros(2 * numTot)

    for i in range(2 * numTot):
        for j in range(5):

            A_coord_y[counter] = i
            A_coord_x[counter] = indexes[i, j]
            A_vals[counter] = weights[i, j]
            counter += 1
        b_coord_y[i] = i
        b_vals[i] = RHS[i][0]

    A = csr_matrix((A_vals, (A_coord_y, A_coord_x)), shape=(2 * numTot, 2 * numTot))
    b = csr_matrix((b_vals, (b_coord_y, np.zeros(2 * numTot))), shape=(2 * numTot, 1))

    x = spsolve(A, b)

    K = np.zeros((N_g, N_g))
    L = np.zeros((N_g, N_g))

    K_dummy = x[:numTot]
    L_dummy = x[numTot:2 * numTot]

    K[np.tril_indices(N_g)] = K_dummy
    L[np.tril_indices(N_g)] = L_dummy
    K = K.transpose()
    L = L.transpose()

    return K, L


def K_solver_2x2(fun, N_g):
    """Wrapper to prepare params and call solver_2x2.

    Inputs:
      fun: object providing mu, lam, lam_d, c_1, c_2, q functions/attributes
      N_g: grid size
    Outputs:
      Kvu, Kvv: kernel matrices returned by solver_2x2
    """
    class Struct:
        pass

    xspan = np.linspace(0, 1, N_g)
    params = Struct()
    params.mu = fun.mu(xspan)
    params.lam = fun.lam(xspan)
    params.a_1 = np.zeros((N_g, N_g))
    params.a_2 = np.zeros((N_g, N_g))
    params.b_1 = np.zeros((N_g, N_g))
    params.b_2 = np.zeros((N_g, N_g))

    for i in range(N_g):
        for j in range(N_g):
            # lambda'(xi)
            params.a_1[i, j] = fun.lam_d(xspan[i])
            # c_2(xi)
            params.a_2[i, j] = fun.c_2(xspan[i])
            # c_1(xi)
            params.b_1[i, j] = fun.c_1(xspan[i])
            # mu'(xi)
            params.b_2[i, j] = -fun.mu(xspan[i])
    # boundary conditions with k(x,x)
    params.f = -fun.c_2(xspan) / (fun.lam(xspan) + fun.mu(xspan))

    # boundary condition with l(x,0)
    params.q = fun.q * fun.lam(0) / fun.mu(0)
    Kvu, Kvv = solver_2x2(params, N_g)
    return Kvu, Kvv
