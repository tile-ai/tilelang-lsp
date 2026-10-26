import torch

from elementwise_add import elementwise_add


def main():
    size = 128
    a = torch.randn(size, size, dtype=torch.float32, device="cuda")
    b = torch.randn(size, size, dtype=torch.float32, device="cuda")
    out = elementwise_add(
        a,
        b,
        block_M=32,
        block_N=32,
        threads=128,
        in_dtype="float32",
        out_dtype="float32",
    )
    torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
    print("TLSP_CUDA_DEBUG_NUMERICS_OK", flush=True)


if __name__ == "__main__":
    main()
