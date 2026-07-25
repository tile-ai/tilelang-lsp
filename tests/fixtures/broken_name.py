# Broken mid-editing kernel: NameError at elaboration (L1 diagnostic path),
# while the static hints for earlier lines must still work.
import tilelang
import tilelang.language as T


# <typecheck> M=1024, block_M=128
@tilelang.jit
def broken_kernel(A, block_M, dtype=T.float16):
    M = T.const("M")

    A: T.Tensor((M,), dtype)
    B = T.empty((M,), dtype)

    with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
        A_shared = T.alloc_shared((block_M,), dtype)
        T.copy(A[bx * block_M], A_shared)
        T.copy(A_shared, B[bx * block_M_typo])  # NameError: block_M_typo

    return B
