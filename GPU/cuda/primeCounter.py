from numba import cuda
import numpy as np
import math
import time


@cuda.jit(device=True)
def is_prime(number):
    if number <= 1:
        return False
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True


@cuda.jit
def prim_numbers(numbers, prime_counter):
    N = numbers.shape[0]
    thread_idx = cuda.grid(1)  # cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x - abs thread position in the grid
    stride = cuda.gridsize(1)  # cuda.blockDim.x * cuda.gridDim.x - how many threads in the grid

    count_local = 0  # v1
    for i in range(int(thread_idx), N, int(stride)):
        if is_prime(numbers[i]):
            count_local += 1  # v1
    cuda.atomic.add(prime_counter, 0, count_local)  # v1


def main():
    N = 2 ** 20

    numbers_host = np.arange(N).astype(np.int32)
    prime_counter_host = np.array([0], dtype=np.int32)

    numbers_device = cuda.to_device(numbers_host)
    prime_counter_device = cuda.to_device(prime_counter_host)

    best_time = float('inf')
    best_G = None  # threads per grid
    best_L = None  # threads per block

    # Testing G i L
    for G in [2 ** i for i in range(10, 15)]:  # grid sizes
        for L in [2 ** j for j in range(5, 8)]:  # block sizes
            blocks = (G + L - 1) // L  # grid size in blocks: number of blocks in one grid

            # reset
            prime_counter_device.copy_to_device(prime_counter_host)

            start_time = time.time()
            prim_numbers[blocks, L](numbers_device, prime_counter_device)
            cuda.synchronize()  # Osiguraj da je kernel izvršen prije mjerenja vremena
            elapsed_time = time.time() - start_time

            print(f"G: {G}, L: {L}, Blocks Per Grid: {blocks} Time: {elapsed_time:.5f} s")

            if elapsed_time < best_time:
                best_time = elapsed_time
                best_G = G
                best_L = L

    prime_counter_host = prime_counter_device.copy_to_host()

    print(f"\n#PRIMEs: {prime_counter_host[0]}")
    print(f"Optimal G: {best_G}, Optimal L: {best_L}, Best Time: {best_time:.5f} s")


if __name__ == "__main__":
    main()
