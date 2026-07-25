# Two kernels in one file: directive selects which one to analyze.
import tilelang
import tilelang.language as T


@tilelang.jit
def kernel_one(A, block_M, dtype=T.float16):
    M = T.const("M")
    A: T.Tensor((M,), dtype)
    B = T.empty((M,), dtype)
    with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
        A_shared = T.alloc_shared((block_M,), dtype)
        T.copy(A[bx * block_M], A_shared)
        T.copy(A_shared, B[bx * block_M])
    return B


# <typecheck> kernel=kernel_two, block_M=64, N=512
@tilelang.jit
def kernel_two(A, block_M, dtype=T.float32):
    N = T.const("N")
    A: T.Tensor((N,), dtype)
    B = T.empty((N,), dtype)
    with T.Kernel(T.ceildiv(N, block_M), threads=64) as bx:
        A_frag = T.alloc_fragment((block_M,), dtype)
        T.copy(A[bx * block_M], A_frag)
        T.copy(A_frag, B[bx * block_M])
    return B
