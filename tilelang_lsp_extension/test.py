import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from functools import partial

name = "Alice"
age = 30
active = True

s_acc=T.alloc_fragment(age,3)

class myClass:
    #age = 1
    A= T.alloc_shared(age)

    def __init__(self, age, 
        active
    ):
        a = 1
        age1= 10
        self.actve =False
        A= T.alloc_shared(age)
    
    def myFunction():
        self.age = 0



o_acc=s_acc
o_acc         =     s_acc
#o_acc=s_acc

block_N = 5
if(1):
    block_N = 10
block_N = 15
block_M = block_N
dtype1 = torch.float16
dtype2 = "bfloat16"

Q_shared = T.alloc_shared(block_M,1)

TestQ_shared = T.alloc_shared([block_M],dtype1)

P_shared=T.alloc_shared([dtype = dtype1])
TestP_shared=T.alloc_shared(dtype2)
def function (
    A:T.buffer(block_M,block_N)
    block=100
):

    #block = 1000
    s_acc
    A; Q_shared[1]
    Q_shared = T.alloc_shared([block)
    
    Q_shared == T.alloc_shared(block)
    dtype = "float"
    block = 999
    Q_shared
    block_N = 123
    Q_shared= T.alloc_shared(block,[ age , block_N], dtype)
    R_shared= T.alloc_shared(block,[ age , block_N], dtype=dtype,scope = "local")

Q_shared

def alloc_var_add(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_var(dtype)
            tmp = 1  # noqa: F841
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = A_shared[i] + tmp
            T.copy(A_shared, B[bx * block_N])

    return main