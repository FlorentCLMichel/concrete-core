#ifndef CNCRT_BSK_H
#define CNCRT_BSK_H

#include "bootstrap.h"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include <atomic>
#include <cstdint>

__device__ inline int get_start_ith_ggsw(int i, uint32_t polynomial_size,
                                         int glwe_dimension,
                                         uint32_t l_gadget) {
    return i * polynomial_size / 2  * (glwe_dimension + 1) * (glwe_dimension + 1) * l_gadget;
}

__device__ double2*
get_ith_mask_kth_block(double2* ptr, int i, int k, int level, uint32_t polynomial_size,
                       int glwe_dimension, uint32_t l_gadget) {
  return &ptr[get_start_ith_ggsw(i, polynomial_size, glwe_dimension, l_gadget) +
          level * polynomial_size / 2 * (glwe_dimension + 1) * (glwe_dimension + 1) +
          k * polynomial_size / 2 * (glwe_dimension + 1)];
}

__device__ double2*
get_ith_body_kth_block(double2 *ptr, int i, int k, int level, uint32_t polynomial_size,
                       int glwe_dimension, uint32_t l_gadget) {
    return &ptr[get_start_ith_ggsw(i, polynomial_size, glwe_dimension, l_gadget) +
                level * polynomial_size / 2 * (glwe_dimension + 1) * (glwe_dimension + 1) +
                k * polynomial_size / 2 * (glwe_dimension + 1) +
                polynomial_size / 2];
}

void cuda_initialize_twiddles(uint32_t polynomial_size, uint32_t gpu_index) {
  cudaSetDevice(gpu_index);
  int sw_size = polynomial_size / 2;
  short *sw1_h, *sw2_h;

  sw1_h = (short *)malloc(sizeof(short) * sw_size);
  sw2_h = (short *)malloc(sizeof(short) * sw_size);

  memset(sw1_h, 0, sw_size * sizeof(short));
  memset(sw2_h, 0, sw_size * sizeof(short));
  int cnt = 0;
  for (int i = 1, j = 0; i < polynomial_size / 2; i++) {
    int bit = (polynomial_size / 2) >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;

    if (i < j) {
      sw1_h[cnt] = i;
      sw2_h[cnt] = j;
      cnt++;
    }
  }
  cudaMemcpyToSymbol(SW1, sw1_h, sw_size * sizeof(short), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(SW2, sw2_h, sw_size * sizeof(short), 0,
                     cudaMemcpyHostToDevice);
  free(sw1_h);
  free(sw2_h);
}

template <typename T, typename ST>
void cuda_convert_lwe_bootstrap_key(double2 *dest, ST *src, void *v_stream,
                               uint32_t gpu_index, uint32_t input_lwe_dim, uint32_t glwe_dim,
                               uint32_t l_gadget, uint32_t polynomial_size) {

  cudaSetDevice(gpu_index);
  int shared_memory_size = sizeof(double) * polynomial_size;

  int total_polynomials =
      input_lwe_dim * (glwe_dim + 1) * (glwe_dim + 1) *
      l_gadget;

  // Here the buffer size is the size of double2 times the number of polynomials
  // times the polynomial size over 2 because the polynomials are compressed
  // into the complex domain to perform the FFT
  size_t buffer_size = total_polynomials * polynomial_size / 2 * sizeof
                       (double2);

  int gridSize = total_polynomials;
  int blockSize = polynomial_size / choose_opt(polynomial_size);
  // todo(Joao): let's use cudaMallocHost here,
  // since it allocates page-staged memory which allows
  // faster data copy
  double2 *h_bsk = (double2 *)malloc(buffer_size);
  double2 *d_bsk;
  cudaMalloc((void **)&d_bsk, buffer_size);

  // compress real bsk to complex and divide it on DOUBLE_MAX
  for (int i = 0; i < total_polynomials; i++) {
    int complex_current_poly_idx = i * polynomial_size / 2;
    int torus_current_poly_idx = i * polynomial_size;
    for (int j = 0; j < polynomial_size / 2; j++) {
      h_bsk[complex_current_poly_idx + j].x =
          src[torus_current_poly_idx + 2 * j];
      h_bsk[complex_current_poly_idx + j].y =
          src[torus_current_poly_idx + 2 * j + 1];
      h_bsk[complex_current_poly_idx + j].x /=
          (double)std::numeric_limits<T>::max();
      h_bsk[complex_current_poly_idx + j].y /=
          (double)std::numeric_limits<T>::max();
    }
  }

  cudaMemcpy(d_bsk, h_bsk, buffer_size, cudaMemcpyHostToDevice);

  auto stream = static_cast<cudaStream_t *>(v_stream);
  switch (polynomial_size) {
    // FIXME (Agnes): check if polynomial sizes are ok
  case 512:
    batch_NSMFFT<FFTDegree<Degree<512>, ForwardFFT>>
    <<<gridSize, blockSize, shared_memory_size, *stream>>>(d_bsk, dest);
    break;
  case 1024:
    batch_NSMFFT<FFTDegree<Degree<1024>, ForwardFFT>>
    <<<gridSize, blockSize, shared_memory_size, *stream>>>(d_bsk, dest);
    break;
  case 2048:
    batch_NSMFFT<FFTDegree<Degree<2048>, ForwardFFT>>
    <<<gridSize, blockSize, shared_memory_size, *stream>>>(d_bsk, dest);
    break;
  case 4096:
    batch_NSMFFT<FFTDegree<Degree<4096>, ForwardFFT>>
    <<<gridSize, blockSize, shared_memory_size, *stream>>>(d_bsk, dest);
    break;
  case 8192:
    batch_NSMFFT<FFTDegree<Degree<8192>, ForwardFFT>>
    <<<gridSize, blockSize, shared_memory_size, *stream>>>(d_bsk, dest);
    break;
  default:
    break;
  }

  cudaFree(d_bsk);
  free(h_bsk);

}

void cuda_convert_lwe_bootstrap_key_32(void *dest, void *src, void *v_stream,
                               uint32_t gpu_index, uint32_t input_lwe_dim, uint32_t glwe_dim,
                               uint32_t l_gadget, uint32_t polynomial_size) {
  cuda_convert_lwe_bootstrap_key<uint32_t, int32_t>((double2 *)dest, (int32_t *)src,
                                           v_stream, gpu_index, input_lwe_dim,
                                           glwe_dim, l_gadget, polynomial_size);
}

void cuda_convert_lwe_bootstrap_key_64(void *dest, void *src, void *v_stream,
                                  uint32_t gpu_index, uint32_t input_lwe_dim, uint32_t glwe_dim,
                                  uint32_t l_gadget, uint32_t polynomial_size) {
  cuda_convert_lwe_bootstrap_key<uint64_t, int64_t>((double2 *)dest, (int64_t *)src,
                                           v_stream, gpu_index, input_lwe_dim,
                                           glwe_dim, l_gadget, polynomial_size);
}

#endif // CNCRT_BSK_H
