import time
import pyopencl as cl
import numpy as np


def jacobistep(psinew, psi, m, n, mf):
    psinew_buffer = cl.Buffer(context, mf.WRITE_ONLY, psinew.nbytes)
    psi_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=psi)

    program.jacobistep(queue, (m, n), None, psinew_buffer, psi_buffer, np.int32(m), np.int32(n))

    cl.enqueue_copy(queue, psinew, psinew_buffer)
    queue.finish()


def deltasq(newarr, oldarr, m, n, mf):
    result = np.zeros_like(newarr, dtype=np.float64)
    result_buffer = cl.Buffer(context, mf.WRITE_ONLY, result.nbytes)
    newarr_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=newarr)
    oldarr_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=oldarr)

    program.deltasq(queue, (m, n), None, newarr_buffer, oldarr_buffer, result_buffer, np.int32(m), np.int32(n))
    cl.enqueue_copy(queue, result, result_buffer)
    queue.finish()

    return np.sum(result)


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


kernel = """
__kernel void jacobistep(__global double *psinew, __global const double *psi, const int m, const int n) {

    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;

    if (i <= m && j <= n) {
        psinew[i*(n+2)+j] = 0.25*(psi[(i-1)*(n+2)+j] + psi[(i+1)*(n+2)+j] + psi[i*(n+2)+j-1] + psi[i*(n+2)+j+1]);
    }
}

__kernel void deltasq(__global const double *newarr, __global const double *oldarr, __global double *result, const int m, const int n) {

    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;

    if (i <= m && j <= n) {
        double temp = (newarr[i*(n+2)+j] - oldarr[i*(n+2)+j]);
        result[i*(n+2)+j] = temp * temp;
    }
}
"""

if __name__ == '__main__':
    printfreq = 50  # output frequency
    tolerance = 0.0

    scalefactor = 64
    numiter = 1000

    bbase = 10  # simulation sizes
    hbase = 15
    wbase = 5
    mbase = 32
    nbase = 32

    checkerr = False

    if tolerance > 0.0:  # do we stop because of tolerance?
        checkerr = True

    b, h, w, m, n = [x * scalefactor for x in [bbase, hbase, wbase, mbase, nbase]]

    psi = np.zeros((m + 2, n + 2), dtype=np.float64)
    psitmp = np.zeros((m + 2, n + 2), dtype=np.float64)

    boundarypsi(psi, m, n, b, h, w)  # set the psi boundary conditions

    bnorm = np.sqrt(np.sum(psi ** 2))  # compute normalisation factor for error

    error = np.inf

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context=context, device=device)
    program = cl.Program(context, kernel).build()
    mf = cl.mem_flags

    time_start = time.time()  # begin Jacobi
    for iter in range(1, numiter + 1):

        jacobistep(psitmp, psi, m, n, mf)  # calculate psi for next iteration

        if checkerr or iter == numiter:  # calculate current error if required
            error = deltasq(psitmp, psi, m, n, mf)
            error = np.sqrt(error)
            error = error / bnorm

        if checkerr:  # quit early if we have reached required tolerance
            if error < tolerance:
                print(f'Converged on iteration {iter}')
                break

        psi[1:m + 1, 1:n + 1] = psitmp[1:m + 1, 1:n + 1]  # copy back

        if iter % printfreq == 0:  # print loop information
            if not checkerr:
                print(f'Completed iteration {iter}')
            else:
                print(f'Completed iteration {iter}, error = {error}')

    time_end = time.time()
    print(f'After {iter} iterations, the error is: {error}')
    print(f"Time for {iter} iterations was {(time_end - time_start):.2f} seconds")
