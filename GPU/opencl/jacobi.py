import time
import numpy as np


def jacobi_step(psi_new, psi, m, n):
    psi_new[1:m + 1, 1:n + 1] = (
            0.25 * (psi[:m, 1:n + 1] + psi[2:m + 2, 1:n + 1] + psi[1:m + 1, :n] + psi[1:m + 1, 2:n + 2]))


def boundary_psi(psi, m, b, h, w):
    for i in range(b + 1, b + w):
        psi[i, 0] = i - b
    for i in range(b + w, m + 1):
        psi[i, 0] = w
    for j in range(1, h + 1):
        psi[m + 1, j] = w
    for j in range(h + 1, h + w):
        psi[m + 1, j] = w - j + h
    return psi


def delta_sq(new_arr, old_arr, m, n):
    dsq = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = (new_arr[i, j] - old_arr[i, j]) ** 2
            dsq += temp
    return dsq


if __name__ == "__main__":

    scale_factor = 64
    num_of_iterations = 1000

    print_freq = 50
    tolerance = 0.0
    check_err = 0

    b_base = 10
    h_base = 15
    w_base = 5
    m_base = 32
    n_base = 32

    b, h, w, m, n = [x * scale_factor for x in [b_base, h_base, w_base, m_base, n_base]]

    psi = np.zeros((m + 2, n + 2))
    psi_tmp = np.zeros((m + 2, n + 2))

    boundary_psi(psi, m, b, h, w)

    b_norm = np.sqrt(np.sum(psi ** 2))
    error = np.inf
    if tolerance > 0.0:
        check_err = 1

    time_start = time.time()

    iter = 0
    for iter in range(1, num_of_iterations + 1):
        jacobi_step(psi_tmp, psi, m, n)

        if check_err or iter == num_of_iterations:
            error = delta_sq(psi_tmp, psi, m, n)
            error = np.sqrt(error)
            error = error / b_norm

        if check_err and error < tolerance:
            print(f'Converged on iteration {iter}')
            break

        psi[1:m + 1, 1:n + 1] = psi_tmp[1:m + 1, 1:n + 1]

        if iter % print_freq == 0:
            if not check_err:
                print(f'Completed iteration {iter}')
            else:
                print(f'Completed iteration {iter}, error = {error}')

    print(f'After {iter} iterations, the error is: {error}')
    time_end = time.time()
    print(f"Time for {iter} iterations was {(time_end - time_start):.2f} seconds")


