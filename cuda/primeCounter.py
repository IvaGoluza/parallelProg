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
            # prime_counter[0] += 1   # v2
            # cuda.atomic.add(prime_counter, 0, 1)  # v3
            count_local += 1   # v1
    cuda.atomic.add(prime_counter, 0, count_local)  # v1


def main():
    N = 2 ** 20
    G = 2 ** 14  # grid size in threads: number of blocks in one grid
    L = 128      # block size: number of threads in one block
    blocks = (G + L - 1) // L  # grid size in blocks: number of blocks in one grid

    numbers_host = np.arange(N).astype(np.int32)
    prime_counter_host = np.array([0], dtype=np.int32)

    numbers_device = cuda.to_device(numbers_host)
    prime_counter_device = cuda.to_device(prime_counter_host)

    start_time = time.time()
    prim_numbers[blocks, L](numbers_device, prime_counter_device)

    prime_counter_host = prime_counter_device.copy_to_host()
    print(f"#PRIMEs: {prime_counter_host}")
    print(f"Time: {time.time() - start_time} s")


if __name__ == "__main__":
    main()
