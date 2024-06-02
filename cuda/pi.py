import time

import numpy as np
from numba import cuda
import math


@cuda.jit
def calculate_pi_par(pi, N):
    thread_idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    h = 1.0 / N[0]
    sum_pi = 0.0

    for i in range(int(thread_idx) + 1, N[0] + 1, int(stride)):
        x = h * (i - 0.5)
        sum_pi += 4.0 / (1.0 + x * x)

    cuda.atomic.add(pi, 0, h * sum_pi)


def calculate_pi_seq(n):
    sum_pi = 0
    for i in range(1, n+1):
        sum_pi += (4/n) / (1+((i-0.5) / n)**2)
    return sum_pi


def main():
    N = 2 ** 28
    G = 2 ** 14  # grid size in threads: number of blocks in one grid
    L = 128  # block size: number of threads in one block
    blocks = (G + L - 1) // L  # grid size in blocks: number of blocks in one grid

    N_device = cuda.to_device(np.array([N], dtype=np.int32))

    pi_device = cuda.to_device(np.array([0], dtype=np.float64))

    start_time = time.time()
    calculate_pi_par[blocks, L](pi_device, N_device)
    # pi_approx = calculate_pi_seq(N)

    pi_approx = np.sum(pi_device.copy_to_host())
    pi_math = math.pi

    print(f"Approximated value of pi: {pi_approx}")
    print(f"Abs error: {abs(pi_approx - pi_math)}")
    print(f"Time: {time.time() - start_time} s")


if __name__ == "__main__":
    main()
