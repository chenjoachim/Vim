#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <ctime>
#include "cuda_fp16.hpp"
#include "cuda_fp16.h"

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"

cudaError_t cutlass_strided_batched_sgemm_int8(
  int m, 
  int n,
  int k,
  const int8_t *A,
  int lda,
  long long int batch_stride_A,
  const int8_t *B,
  int ldb,
  long long int batch_stride_B,
  int32_t *C,
  int ldc,
  long long int batch_stride_C,
  int batch_count) {

  using ElementComputeEpilogue = int32_t;
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); // alpha = 1
  ElementComputeEpilogue beta = ElementComputeEpilogue(0); // beta = 0 (default GEMM config)

  using Gemm = cutlass::gemm::device::GemmBatched<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

cudaError_t cutlass_strided_batched_sgemm_int4(
  int m, 
  int n,
  int k,
  const int8_t *A, // const cutlass::int4b_t *A
  int lda,
  long long int batch_stride_A,
  const int8_t *B, // const cutlass::int4b_t *B
  int ldb,
  long long int batch_stride_B,
  int32_t *C,
  int ldc,
  long long int batch_stride_C,
  int batch_count) {

  using ElementComputeEpilogue = int32_t;
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); // alpha = 1
  ElementComputeEpilogue beta = ElementComputeEpilogue(0); // beta = 0 (default GEMM config)

  using Gemm = cutlass::gemm::device::GemmBatched<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

// activation smoothing and quantization - per tensor (asymmetric with zero point)
template<typename scalar_t>
__global__ void act_smq_per_tensor(scalar_t * MatI, int8_t * MatO, scalar_t * smooth_scales, scalar_t * quant_scales, scalar_t * zero_points, int qbit, int rows, int cols){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Row >= rows || Col >= cols) return;

  int matIdx = Row * cols + Col;
  scalar_t val = MatI[matIdx];
  scalar_t smooth_scale = smooth_scales[Col];
  scalar_t quant_scale = quant_scales[0];
  scalar_t z = zero_points[0];
  if (qbit == 8){
    int q = (int)round((float)(val / (quant_scale * smooth_scale)) + (float)z);
    q = std::clamp(q, 0, 255);
    MatO[matIdx] = (int8_t)(q - 128);  // shift to signed range for int8 GEMM
  }
  else if (qbit == 4) {
    int q = (int)round((float)(val / (quant_scale * smooth_scale)) + (float)z);
    q = std::clamp(q, 0, 15);
    MatO[matIdx] = (int8_t)q;  // [0,15] fits in signed int8
  }
}

// activation smoothing and quantization - per token (asymmetric with zero point)
template<typename scalar_t>
__global__ void act_smq_per_token(scalar_t * MatI, int8_t * MatO, scalar_t * smooth_scales, scalar_t * quant_scales, scalar_t * zero_points, int qbit, int rows, int cols){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Row >= rows || Col >= cols) return;

  int matIdx = Row * cols + Col;
  scalar_t val = MatI[matIdx];
  scalar_t smooth_scale = smooth_scales[Col];
  scalar_t quant_scale = quant_scales[Row];
  scalar_t z = zero_points[Row];
  if (qbit == 8){
    int q = (int)round((float)(val / (quant_scale * smooth_scale)) + (float)z);
    q = std::clamp(q, 0, 255);
    MatO[matIdx] = (int8_t)(q - 128);
  }
  else if (qbit == 4) {
    int q = (int)round((float)(val / (quant_scale * smooth_scale)) + (float)z);
    q = std::clamp(q, 0, 15);
    MatO[matIdx] = (int8_t)q;
  }
}

// dequant with zero-point correction - per tensor
// y = s * (gemm + zp_shift * w_col_sum[Col])
// where zp_shift = 128 - z (8-bit) or -z (4-bit)
template<typename scalar_t>
__global__ void dequantize_per_tensor(const int32_t * gemm, scalar_t * __restrict__ output, scalar_t * s, scalar_t * w_col_sum, scalar_t * zp_shift, int rows, int cols){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Row >= rows || Col >= cols) return;

  int matIdx = Row * cols + Col;
  output[matIdx] = s[0] * ((scalar_t)gemm[matIdx] + zp_shift[0] * w_col_sum[Col]);
}

// dequant with zero-point correction - per token
// y = s[Row,Col] * (gemm + zp_shift[Row] * w_col_sum[Col])
template<typename scalar_t>
__global__ void dequantize_per_token(const int32_t * gemm, scalar_t * __restrict__ output, scalar_t * s, scalar_t * w_col_sum, scalar_t * zp_shift, int rows, int cols){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Row >= rows || Col >= cols) return;

  int matIdx = Row * cols + Col;
  output[matIdx] = s[matIdx] * ((scalar_t)gemm[matIdx] + zp_shift[Row] * w_col_sum[Col]);
}

torch::Tensor vim_GEMM_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor smooth_scale, torch::Tensor scale_x, torch::Tensor scale_w, torch::Tensor z, torch::Tensor w_col_sum, int H_size, int qbit){
  cudaError_t result;

  //x = (batch_count, m, k): fp16, w = (n, k): int8, GEMM = (batch_count, m, n): int32
  //For int4, w = (n, k//2): int8
  //smooth_scale = (1, k): fp16, scale_x = (m, 1): fp16, scale_w = (n, 1): fp16 - per token
  //smooth_scale = (1, k): fp16, scale_x = (1, 1): fp16, scale_w = (1, 1): fp16 - per tensor
  //z = zero point for asymmetric act quant, same shape as scale_x
  //w_col_sum = (n,): sum of int weight rows, for zero-point correction in dequant
  long long int m = x.size(1);
  long long int k = x.size(2);
  long long int n = w.size(0);
  long long int batched_count = x.size(0);
  long long int x_height = batched_count * m;
  int block_height = H_size;
  int block_width = 1024/H_size;

  torch::Tensor x_colm = x.view({batched_count * m, k}).contiguous();

  dim3 block_size(block_width, block_height);  // (64, 16) = 1024 threads
  // grid: (ceil(cols/block_width), ceil(rows/block_height))
  dim3 grid_size((k + block_width - 1) / block_width, (x_height + block_height - 1) / block_height);
  auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
  torch::Tensor gemm = torch::empty({batched_count * m, n}, option_gemm);

  // Compute zp_shift for dequant correction: 128 - z (8-bit) or -z (4-bit)
  int shift = (qbit == 8) ? 128 : 0;
  torch::Tensor zp_shift = (shift - z).to(x.dtype()).contiguous();

  if (qbit == 8){
    // activation smoothing and quantization (asymmetric)
    auto option_x_q = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    torch::Tensor x_q = torch::empty({x_height, k}, option_x_q);

    if (scale_x.numel() == 1){
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_tensor<scalar_t><<<grid_size, block_size>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(), qbit, x_height, k);
      }));
    }
    else{
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_token<scalar_t><<<grid_size, block_size>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(), qbit, x_height, k);
      }));
    }

    // GEMM
    int const lda = k;
    int const ldb = k;
    int const ldc = n;

    long long int batch_stride_A = static_cast<long long int>(lda) * m;
    long long int batch_stride_B = 0;
    long long int batch_stride_C = static_cast<long long int>(ldc) * m;

    result = cutlass_strided_batched_sgemm_int8(m, n, k,
            x_q.data_ptr<int8_t>(), lda, batch_stride_A,
            w.data_ptr<int8_t>(), ldb, batch_stride_B,
            gemm.data_ptr<int32_t>(), ldc, batch_stride_C, batched_count);
  }
  else if (qbit == 4) {
    // activation smoothing and quantization (asymmetric)
    long long int k_int4 = k>>1;

    auto option_x_q = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    torch::Tensor x_q = torch::empty({x_height, k_int4}, option_x_q);

    // For int4, grid covers k_int4 columns (each int8 stores one 4-bit value)
    dim3 grid_size_int4((k_int4 + block_width - 1) / block_width, (x_height + block_height - 1) / block_height);

    if (scale_x.numel() == 1){
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_tensor<scalar_t><<<grid_size_int4, block_size>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(), qbit, x_height, k_int4);
      }));
    }
    else{
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_token<scalar_t><<<grid_size_int4, block_size>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), z.data_ptr<scalar_t>(), qbit, x_height, k_int4);
      }));
    }

    // GEMM
    int const lda = k_int4;
    int const ldb = k_int4;
    int const ldc = n;

    long long int batch_stride_A = static_cast<long long int>(lda) * m;
    long long int batch_stride_B = 0;
    long long int batch_stride_C = static_cast<long long int>(ldc) * m;

    result = cutlass_strided_batched_sgemm_int4(m, n, k_int4,
            x_q.data_ptr<int8_t>(), lda, batch_stride_A,
            w.data_ptr<int8_t>(), ldb, batch_stride_B,
            gemm.data_ptr<int32_t>(), ldc, batch_stride_C, batched_count);
  }

  // Dequant with zero-point correction
  // y = final_scale * (gemm + zp_shift * w_col_sum)
  auto option_dequant = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  torch::Tensor y = torch::empty({batched_count * m, n}, option_dequant);

  dim3 grid_size_dequant((n + block_width - 1) / block_width, (x_height + block_height - 1) / block_height);

  if (scale_x.numel() == 1){
    torch::Tensor final_scale = scale_x * scale_w;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    dequantize_per_tensor<scalar_t><<<grid_size_dequant, block_size>>>(
        gemm.data_ptr<int32_t>(),
        y.data_ptr<scalar_t>(),
        final_scale.data_ptr<scalar_t>(),
        w_col_sum.data_ptr<scalar_t>(),
        zp_shift.data_ptr<scalar_t>(),
        x_height, n);
    }));
  }
  else{
    torch::Tensor final_scale = torch::matmul(scale_x, scale_w.transpose(0,1));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    dequantize_per_token<scalar_t><<<grid_size_dequant, block_size>>>(
        gemm.data_ptr<int32_t>(),
        y.data_ptr<scalar_t>(),
        final_scale.data_ptr<scalar_t>(),
        w_col_sum.data_ptr<scalar_t>(),
        zp_shift.data_ptr<scalar_t>(),
        x_height, n);
    }));
  }

  return y.view({batched_count, m, n}).contiguous();

}