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
    for i in range(1, n + 1):
        sum_pi += (4 / n) / (1 + ((i - 0.5) / n) ** 2)
    return sum_pi


def measure_time_seq(n):
    start_time = time.time()
    pi_approx = calculate_pi_seq(n)
    end_time = time.time()
    return pi_approx, end_time - start_time


def measure_time_par(N, G, L):
    start_time = time.time()
    blocks = (G + L - 1) // L
    N_device = cuda.to_device(np.array([N], dtype=np.int32))
    pi_device = cuda.to_device(np.array([0], dtype=np.float64))

    calculate_pi_par[blocks, L](pi_device, N_device)
    cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()

    pi_approx = np.sum(pi_device.copy_to_host())
    elapsed_time = end_time - start_time
    return pi_approx, elapsed_time


def main():
    N = 2 ** 20  # Reduced N for quicker testing; use larger value in production
    grid_sizes = [2 ** i for i in range(10, 15)]
    block_sizes = [2 ** j for j in range(5, 8)]

    # Measure sequential computation time
    pi_seq, T1 = measure_time_seq(N)
    pi_math = math.pi
    print(f"Sequential - Approximated value of pi: {pi_seq}, Abs error: {abs(pi_seq - pi_math)}, Time: {T1} s")

    best_time = float('inf')
    best_G = None
    best_L = None
    best_speedup = 0

    for G in grid_sizes:
        for L in block_sizes:
            pi_par, TP = measure_time_par(N, G, L)
            if TP == 0.0:
                print(f"Skipped M: {N/G}, L: {L} due to zero elapsed time")
                continue
            P = L
            speedup = T1 / TP
            efficiency = speedup / P

            print(f"M: {N/G}, L: {L}, TP: {TP}, Speedup: {speedup}, Efficiency: {efficiency}")

            if TP < best_time:
                best_time = TP
                best_G = G
                best_L = L
                best_speedup = speedup

    print(f"\nOptimal Parameters - G: {best_G}, L: {best_L}")
    print(f"Optimal Speedup: {best_speedup}, Best Time: {best_time} s")


if __name__ == "__main__":
    main()
