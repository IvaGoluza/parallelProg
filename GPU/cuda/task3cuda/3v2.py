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
checkerr = 0


@cuda.jit
def jacobistep(psinew, psi, m, n):
    i, j = cuda.grid(2)
    if i < m[0] and j < n[0]:
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


scalefactor = 64
numiter = 1000
time_start = time.time()

b, h, w, m, n = [x * scalefactor for x in [bbase, hbase, wbase, mbase, nbase]]

psi = np.zeros((m + 2, n + 2))
psitmp = np.zeros((m + 2, n + 2))
np.set_printoptions(threshold=sys.maxsize)

boundarypsi(psi, m, n, b, h, w)

bnorm = np.sqrt(np.sum(psi ** 2))
error = np.inf
if tolerance > 0.0:
    checkerr = 1
m_d = cuda.to_device(np.array([m], dtype=np.int32))
n_d = cuda.to_device(np.array([n], dtype=np.int32))

# Allocate device memory once
psitmp_d = cuda.to_device(psitmp)
psi_d = cuda.to_device(psi)

threadsperblock = (32, 32)
blockspergrid_x = (m + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

for iter in range(1, numiter + 1):
    jacobistep[blockspergrid, threadsperblock](psitmp_d, psi_d, m_d, n_d)
    # cuda.synchronize()

    if checkerr or iter == numiter:
        psitmp = psitmp_d.copy_to_host()
        psi = psi_d.copy_to_host()
        error = deltasq(psitmp, psi, m, n)
        error = np.sqrt(error)
        error = error / bnorm

    if checkerr and error < tolerance:
        break

    psi[1:m + 1, 1:n + 1] = psitmp[1:m + 1, 1:n + 1]
    # psitmp_d.copy_to_device(psi_d)
    # psi_d.copy_to_device(psitmp_d)

    if iter % printfreq == 0:
        if not checkerr:
            print(f'Completed {iter} iterations')
        else:
            print(f'Completed {iter} iterations, error = {error:.8f}')


print(f'After {iter} iterations, error: {error:.8f}')
time_end = time.time()
print(f"Time: {(time_end - time_start):.2f}s")
