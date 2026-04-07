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
#include "cutlass/integer_subbyte.h"

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
    int32_t, cutlass::layout::RowMajor,
    int32_t,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,              // Use tensor cores
    cutlass::arch::Sm80                          // Ampere (compatible with Sm86)
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
  int k,  // logical k (number of int4 elements, NOT packed bytes)
  const cutlass::int4b_t *A,
  int lda,
  long long int batch_stride_A,
  const cutlass::int4b_t *B,
  int ldb,
  long long int batch_stride_B,
  int32_t *C,
  int ldc,
  long long int batch_stride_C,
  int batch_count) {

  using ElementComputeEpilogue = int32_t;
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  using Gemm = cutlass::gemm::device::GemmBatched<
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,              // Use tensor cores (required for int4)
    cutlass::arch::Sm80,                         // Ampere (compatible with Sm86)
    cutlass::gemm::GemmShape<128, 128, 128>,     // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 128>,       // WarpShape (K=128 so K/InstructionK=2)
    cutlass::gemm::GemmShape<16, 8, 64>          // InstructionShape (m16n8k64 for int4)
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
    MatO[matIdx] = (int8_t)(q - 8);  // shift to signed range [-8,7] for int4 GEMM
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
    MatO[matIdx] = (int8_t)(q - 8);  // shift to signed range [-8,7] for int4 GEMM
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

// Pack two int8 values (each holding a 4-bit value in [0,15] or [-8,7]) into packed int4b_t pairs.
// Input: (rows, cols) int8 tensor where cols is even.  Output: (rows, cols/2) uint8 tensor
// Each output byte holds two int4 values: low nibble = src[2*i], high nibble = src[2*i+1]
// This matches CUTLASS int4b_t RowMajor packing order.
__global__ void pack_int4_kernel(const int8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, int cols){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * (cols / 2);
  if (idx >= total) return;

  int row = idx / (cols / 2);
  int packed_col = idx % (cols / 2);
  int src_idx = row * cols + packed_col * 2;

  uint8_t lo = (uint8_t)(src[src_idx]     & 0x0F);
  uint8_t hi = (uint8_t)(src[src_idx + 1] & 0x0F);
  dst[idx] = lo | (hi << 4);
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
  int shift = (qbit == 8) ? 128 : 8;  // 128 for int8, 8 for int4 (matching the q-shift in act quant)
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
    // int4 quantization with real int4 GEMM via CUTLASS int4b_t
    // Step 1: quantize activations to int8 (each value in [0,15])
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

    // Step 2: pack int8 activations into int4 pairs (two values per byte)
    int packed_k = k / 2;
    auto option_packed = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    torch::Tensor x_packed = torch::empty({x_height, packed_k}, option_packed);

    int total_packed = x_height * packed_k;
    int pack_threads = 256;
    int pack_blocks = (total_packed + pack_threads - 1) / pack_threads;
    pack_int4_kernel<<<pack_blocks, pack_threads>>>(
        x_q.data_ptr<int8_t>(), x_packed.data_ptr<uint8_t>(), x_height, k);

    // Step 3: real int4 GEMM
    // lda/ldb are logical k (number of int4 elements), not packed bytes
    int const lda = k;
    int const ldb = k;
    int const ldc = n;

    // batch strides in units of int4 elements
    long long int batch_stride_A = static_cast<long long int>(lda) * m;
    long long int batch_stride_B = 0;
    long long int batch_stride_C = static_cast<long long int>(ldc) * m;

    // w is already packed as int4 from Python side (shape [n, k/2] uint8)
    result = cutlass_strided_batched_sgemm_int4(m, n, k,
            reinterpret_cast<const cutlass::int4b_t*>(x_packed.data_ptr<uint8_t>()), lda, batch_stride_A,
            reinterpret_cast<const cutlass::int4b_t*>(w.data_ptr<uint8_t>()), ldb, batch_stride_B,
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