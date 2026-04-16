import os
import sys

import torch
import numpy as np
import time
import statistics
from tqdm import tqdm
import vim_GEMM

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

eps = 1e-8

b = 256
m = 197 # 197
k = 768 # 768 # 1536
n = 3072 # 3072 # 80

quantBits = 4 # 4 or 8
testTurn = 200

def int_quantizer_BWD(x, quantBits):
    '''
    original
    '''
    mx = x.abs().max()
    scale = mx / (2**(quantBits-1)-1)
    x = torch.clamp(x, -mx, mx) / (scale+eps)
    x_int = torch.round(x)
    return x_int

def main():

    FP_time = []
    kernel_time = []

    A = torch.Tensor(np.random.normal(1., 0.1, b*m*k)).reshape(b, m, k).to('cuda')
    B = torch.Tensor(np.random.normal(1., 0.1, n*k)).reshape(n, k).to('cuda')
    smooth_scale = torch.Tensor(np.random.normal(1., 0.1, k)).to('cuda')

    if quantBits == 4:
        B_int4 = torch.Tensor(np.random.normal(1., 0.1, n*k//2)).reshape(n, k//2).to('cuda')
        smooth_scale = torch.Tensor(np.random.normal(1., 0.1, k)).to('cuda')  # must be size k, not k//2

    a_s = torch.Tensor(np.random.normal(1., 0.1, b * m)).reshape(b * m, 1).to('cuda')
    w_s = torch.Tensor(np.random.normal(1., 0.1, n)).reshape(n, 1).to('cuda')

    # zero points (same shape as a_s) and weight column sums (shape n) for zero-point correction
    z = torch.zeros(b * m, 1).to('cuda')
    w_col_sum = torch.zeros(n).to('cuda')

    B_q = int_quantizer_BWD(B_int4, quantBits) if quantBits == 4 else int_quantizer_BWD(B, quantBits)

    for i in tqdm(range(testTurn)):

        start = time.time()
        gemm_fake = (A @ B.mT)
        torch.cuda.synchronize()
        FP_time.append(time.time() - start)

        # int4 kernel expects uint8 (packed int4); int8 kernel expects int8
        if quantBits == 4:
            B_q_w = B_q.to(torch.uint8).contiguous()
        else:
            B_q_w = B_q.type(torch.int8).contiguous()
        A = A.contiguous()
        start = time.time()
        gemm_real = vim_GEMM.vim_GEMM(A, \
                    B_q_w, \
                    smooth_scale, \
                    a_s, \
                    w_s, \
                    z, \
                    w_col_sum, \
                    16,
                    quantBits)
        torch.cuda.synchronize()
        kernel_time.append(time.time() - start)


    print(f"{statistics.median(FP_time):.8f} nsec | FP")
    print(f"{statistics.median(kernel_time):.8f} nsec | kernel")

if __name__ == '__main__':
    main()