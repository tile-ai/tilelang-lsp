# Torch-free elementwise kernel for integration tests (target="c" friendly).
import tilelang
import tilelang.language as T


@tilelang.jit
def elementwise_add(A, B, block_M, block_N, in_dtype, out_dtype, threads):
    M, N = T.const("M, N")

    A: T.Tensor((M, N), in_dtype)
    B: T.Tensor((M, N), in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), in_dtype)
        B_shared = T.alloc_shared((block_M, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        T.copy(A[by * block_M, bx * block_N], A_shared)
        T.copy(B[by * block_M, bx * block_N], B_shared)
        for local_y, local_x in T.Parallel(block_M, block_N):
            C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


# <typecheck> M=1024, N=1024, block_M=32, block_N=32, threads=128, in_dtype="float32", out_dtype="float32"
