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
    start = cuda.grid(1)
    for i in range(start, m[0] + 1):
        for j in range(start, n[0] + 1):
            rez = 0.25 * (psi[(i - 1) * (m[0] + 2) + j] + psi[(i + 1) * (m[0] + 2) + j]
                          + psi[i * (m[0] + 2) + j - 1] + psi[i * (m[0] + 2) + j + 1])
            cuda.atomic.add(psinew, i * (m[0] + 2) + j, rez)

@cuda.jit
def copy_array(a, b):
    a[0] = b[0]


def deltasq(newarr, oldarr, m, n):
    dsq = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = (newarr[i, j] - oldarr[i, j]) ** 2
            dsq += temp
    return dsq


def boundarypsi(psi, m, n, b, h, w):
    # BCs on bottom edge

    for i in range(b + 1, b + w):
        psi[i * (m + 2) + 0] = float(i - b)

    for i in range(b + w, m + 1):
        psi[i * (m + 2) + 0] = float(w)

    # BCS on RHS

    for j in range(1, h + 1):
        psi[(m + 1) * (m + 2) + j] = float(w)

    for j in range(h + 1, h + w):
        psi[(m + 1) * (m + 2) + j] = float(w - j + h)


scalefactor = 64
numiter = 1000
time_start = time.time()

b, h, w, m, n = [x * scalefactor for x in [bbase, hbase, wbase, mbase, nbase]]

psi = np.zeros((m+2) * (n+2), dtype=float)
psitmp = np.zeros((m+2) * (n+2), dtype=float)

for i in range(0, m + 2):
    for j in range(0, n + 2):
        psi[i * (m + 2) + j] = 0.0

np.set_printoptions(threshold=sys.maxsize)

boundarypsi(psi, m, n, b, h, w)

bnorm = np.sqrt(np.sum(psi ** 2))
error = np.inf
if tolerance > 0.0:
    checkerr = 1
m_d = cuda.to_device(np.array([m], dtype=np.int32))
n_d = cuda.to_device(np.array([n], dtype=np.int32))

psitmp_d = cuda.to_device(psitmp)  # Dodano
psi_d = cuda.to_device(psi)

threadsperblock = (16, 16)
blockspergrid_x = (m + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

for iter in range(1, numiter + 1):

    jacobistep[blockspergrid, threadsperblock](psitmp_d, psi_d, m_d, n_d)  # Izmijenjeno
    # cuda.synchronize()

    psitmp = psitmp_d.copy_to_host()  # Izmijenjeno
    psi = psi_d.copy_to_host()

    if checkerr or iter == numiter:
        error = deltasq(psitmp, psi, m, n)
        error = np.sqrt(error)
        error = error / bnorm

    if checkerr:
        if error < tolerance:
            print(f'Konvergirano u iteraciji {iter}')
            break

    copy_array[blockspergrid, threadsperblock](psi_d, psitmp_d)

    if iter % printfreq == 0:
        if not checkerr:
            print(f'Completed iterations {iter}')
        else:
            print(f'Completed iterations {iter}, error = {error:.8f}')

print(f'After {iter} iteration, error: {error:.8f}')
time_end = time.time()
print(f"Time: {(time_end - time_start):.2f}s")
