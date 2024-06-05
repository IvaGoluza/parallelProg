import sys
import time
import numpy as np
from numba import cuda

printfreq = 100
tolerance = 0.0
bbase = 10
hbase = 15
wbase = 5
mbase = 32
nbase = 32
checkerror = 0

scalefactor = 64
numiter = 1000

@cuda.jit
def jacobistep(psinew, psi, m, n):
    i, j = cuda.grid(2)
    i += 1
    j += 1
    if i <= m[0] and j <= n[0]:
        psinew[i, j] = 0.25 * (psi[i - 1, j] + psi[i + 1, j] + psi[i, j - 1] + psi[i, j + 1])


def deltasq(newarr, oldarr, m, n):
    dsq = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = (newarr[i, j] - oldarr[i, j]) ** 2
            dsq += temp
    return dsq


def boundarypsi(psi, m, n, b, h, w):
    for i in range(b + 1, b + w):
        psi[i, 0] = i - b
    for i in range(b + w, m + 1):
        psi[i, 0] = w
    for j in range(1, h + 1):
        psi[m + 1, j] = w
    for j in range(h + 1, h + w):
        psi[m + 1, j] = w - j + h
    return psi


if __name__ == "__main__":
    time_start = time.time()
    b = bbase * scalefactor
    h = hbase * scalefactor
    w = wbase * scalefactor
    m = mbase * scalefactor
    n = nbase * scalefactor

    psi = np.zeros((m + 2, n + 2), dtype=np.float64)
    psitmp = np.zeros((m + 2, n + 2), dtype=np.float64)
    np.set_printoptions(threshold=sys.maxsize)

    boundarypsi(psi, m, n, b, h, w)

    bnorm = np.sqrt(np.sum(psi ** 2))
    error = np.inf
    if tolerance > 0.0:
        checkerror = True

    threadsperblock = (32, 32)
    blockspergrid_x = (m + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_psitmp = cuda.to_device(psitmp)
    d_psi = cuda.to_device(psi)
    d_m = cuda.to_device(np.array([m], dtype=np.int32))
    d_n = cuda.to_device(np.array([n], dtype=np.int32))

    for iter in range(1, numiter + 1):
        jacobistep[blockspergrid, threadsperblock](d_psitmp, d_psi, d_m, d_n)
        cuda.synchronize()

        if checkerror or iter == numiter:
            psitmp = d_psitmp.copy_to_host()
            psi = d_psi.copy_to_host()
            error = deltasq(psitmp, psi, m, n)
            error = np.sqrt(error)
            error = error / bnorm

        if checkerror and error < tolerance:
            break

        psi[1:m + 1, 1:n + 1] = psitmp[1:m + 1, 1:n + 1]
        d_psi = cuda.to_device(psi)
        d_psitmp = cuda.to_device(psitmp)

        if iter % printfreq == 0:
            if not checkerror:
                print(f'Completed {iter} iterations')
            else:
                print(f'Completed {iter} iterations, error = {error:.8f}')

    print(f'After {iter} iteration, error: {error:.8f}')
    time_end = time.time()
    print(f"Time: {(time_end - time_start):.2f}s")
