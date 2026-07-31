# Kernel with a genuine data race: every iteration of the T.Parallel loop
# writes C_local[0, 0] with a different value. VerifyParallelLoop aggregates
# it into one warning carrying the store's span (warning diagnostic path).
import tilelang
import tilelang.language as T


@tilelang.jit
def racy_copy(A, B, block_M, block_N, dtype=T.float16):
    M, N = T.const("M, N")

    A: T.Tensor((M, N), dtype)
    B: T.Tensor((M, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), dtype)
        T.copy(A[by * block_M, bx * block_N], A_shared)
        for local_y, local_x in T.Parallel(block_M, block_N):
            C_local[0, 0] = A_shared[local_y, local_x]  # race: same cell, all iterations
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# <typecheck> kernel=racy_copy, M=1024, N=1024, block_M=32, block_N=32, dtype="float16"
