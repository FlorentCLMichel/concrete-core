#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

#ifndef CNCRT_AMORTIZED_PBS_H
#define CNCRT_AMORTIZED_PBS_H

#include "cooperative_groups.h"

#include "../include/helper_cuda.h"
#include "bootstrap.h"
#include "complex/operations.cuh"
//#include "crypto/bootstrapping_key.cuh"
#include "crypto/gadget.cuh"
#include "crypto/torus.cuh"
#include "fft/bnsmfft.cuh"
#include "fft/smfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial.cuh"
#include "polynomial/polynomial_math.cuh"
#include "utils/memory.cuh"
#include "utils/timer.cuh"

template <typename Torus, class params, sharedMemDegree SMD>
/*
 * Kernel launched by host_bootstrap_amortized
 *
 * Uses shared memory to increase performance
 *  - lwe_out: output batch of num_samples bootstrapped ciphertexts c =
 * (a0,..an-1,b) where n is the LWE dimension
 *  - lut_vector: should hold as many test vectors of size polynomial_size
 * as there are input ciphertexts, but actually holds
 * num_lut_vectors vectors to reduce memory usage
 *  - lut_vector_indexes: stores the index corresponding to which test vector
 * to use for each sample in lut_vector
 *  - lwe_in: input batch of num_samples LWE ciphertexts, containing n mask
 * values + 1 body value
 *  - bootstrapping_key: RGSW encryption of the LWE secret key sk1 under secret
 * key sk2
 *  - device_mem: pointer to the device's global memory in case we use it (SMD
 * == NOSM or PARTIALSM)
 *  - lwe_mask_size: size of the Torus vector used to encrypt the input
 * LWE ciphertexts - referred to as n above (~ 600)
 *  - polynomial_size: size of the test polynomial (test vector) and size of the
 * GLWE polynomial (~1024)
 *  - base_log: log base used for the gadget matrix - B = 2^base_log (~8)
 *  - l_gadget: number of decomposition levels in the gadget matrix (~4)
 *  - gpu_num: index of the current GPU (useful for multi-GPU computations)
 *  - lwe_idx: equal to the number of samples per gpu x gpu_num
 *  - device_memory_size_per_sample: amount of global memory to allocate if SMD
 * is not FULLSM
 */
__global__ void device_bootstrap_amortized(
    Torus *lwe_out,
    Torus *lut_vector,
    uint32_t *lut_vector_indexes,
    Torus *lwe_in,
    double2 *bootstrapping_key,
    char *device_mem,
    uint32_t lwe_mask_size,
    uint32_t polynomial_size,
    uint32_t base_log,
    uint32_t l_gadget,
    uint32_t lwe_idx,
    size_t device_memory_size_per_sample) {
  // We use shared memory for the polynomials that are used often during the
  // bootstrap, since shared memory is kept in L1 cache and accessing it is
  // much faster than global memory
  extern __shared__ char sharedmem[];
  char *selected_memory;

  if constexpr (SMD == FULLSM)
    selected_memory = sharedmem;
  else
    selected_memory = &device_mem[blockIdx.x * device_memory_size_per_sample];

  // For GPU bootstrapping the RLWE dimension is hard-set to 1: there is only
  // one mask polynomial and 1 body to handle Also, since the decomposed
  // polynomials take coefficients between -B/2 and B/2 they can be represented
  // with only 16 bits, assuming the base log does not exceed 2^16
  int16_t *accumulator_mask_decomposed = (int16_t *)selected_memory;
  // TODO (Agnes) why not the 16 bits representation here?
  int16_t *accumulator_body_decomposed =
      (int16_t *)accumulator_mask_decomposed + polynomial_size;
  Torus *accumulator_mask = (Torus *)accumulator_body_decomposed +
                            polynomial_size / (sizeof(Torus) / sizeof(int16_t));
  Torus *accumulator_body =
      (Torus *)accumulator_mask + (ptrdiff_t)polynomial_size;
  Torus *accumulator_mask_rotated =
      (Torus *)accumulator_body + (ptrdiff_t)polynomial_size;
  Torus *accumulator_body_rotated =
      (Torus *)accumulator_mask_rotated + (ptrdiff_t)polynomial_size;
  double2 *mask_res_fft = (double2 *)accumulator_body_rotated +
                          polynomial_size / (sizeof(double2) / sizeof(Torus));
  double2 *body_res_fft =
      (double2 *)mask_res_fft + (ptrdiff_t)polynomial_size / 2;
  double2 *accumulator_fft = (double2 *)sharedmem;
  if constexpr (SMD != PARTIALSM)
    accumulator_fft =
        (double2 *)body_res_fft + (ptrdiff_t)(polynomial_size / 2);

  /*
  int dif0 = ((char*)accumulator_body_decomposed - (char*)selected_memory);
  int dif1 = ((char*)accumulator_mask - (char*)accumulator_body_decomposed);
  int dif2 = ((char*)accumulator_body - (char*)accumulator_mask);
  int dif3 = ((char*)accumulator_mask_rotated - (char*)accumulator_body);
  int dif4 = ((char*)accumulator_body_rotated -
  (char*)accumulator_mask_rotated); int dif5 = ((char*)mask_res_fft -
  (char*)accumulator_body_rotated); int dif6 = ((char*)body_res_fft -
  (char*)mask_res_fft); int dif7 =  (SMD != PARTIALSM)? (char*)accumulator_fft -
  (char*)body_res_fft:0; if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("device and shared mem: %d %d %d %d %d %d %d %d\n ",dif0, dif1, dif2,
  dif3, dif4, dif5, dif6, dif7);
  }
  */

  auto block_lwe_in = &lwe_in[blockIdx.x * (lwe_mask_size + 1)];
  Torus *block_lut_vector =
      &lut_vector[lut_vector_indexes[lwe_idx + blockIdx.x] * params::degree * 2];

  // TODO (Agnes) try to store the gadget matrix in const memory to see if
  // register use decreases Since all const mem is used for twiddles currently,
  // it would mean moving some of them to global memory instead
  GadgetMatrix<Torus, params> gadget(base_log, l_gadget);

  // Put "b", the body, in [0, 2N[
  Torus b_hat = rescale_torus_element(
      block_lwe_in[lwe_mask_size],
      2 * params::degree); // 2 * params::log2_degree + 1);

  divide_by_monomial_negacyclic_inplace<Torus, params::opt,
      params::degree / params::opt>(
      accumulator_mask, block_lut_vector, b_hat, false);

  divide_by_monomial_negacyclic_inplace<Torus, params::opt,
      params::degree / params::opt>(
      accumulator_body, &block_lut_vector[params::degree], b_hat, false);

  // Loop over all the mask elements of the sample to accumulate
  // (X^a_i-1) multiplication, decomposition of the resulting polynomial
  // into l_gadget polynomials, and performing polynomial multiplication
  // via an FFT with the RGSW encrypted secret key
  for (int iteration = 0; iteration < lwe_mask_size; iteration++) {
    // TODO make sure that following sync is necessary
    synchronize_threads_in_block();

    // Put "a" in [0, 2N[ instead of Zq
    Torus a_hat = rescale_torus_element(
        block_lwe_in[iteration],
        2 * params::degree); // 2 * params::log2_degree + 1);

    // TODO (Agnes) why is there this if condition?
    if (a_hat == 0) {
      // todo(Joao): **cannot use this optimization**
      // the reason is that one of the input ciphertexts (blockIdx.z)
      // might skip an iteration while others don't, which as a result
      // will make that block not call the grid.sync(), causing a deadlock;
      // maybe it's a workaround to add grid.sync() here, but not sure if
      // there are any edge cases?

      // continue
    }

    // Perform ACC * (X^ä - 1)
    multiply_by_monomial_negacyclic_and_sub_polynomial<
        Torus, params::opt, params::degree / params::opt>(
        accumulator_mask, accumulator_mask_rotated, a_hat);

    multiply_by_monomial_negacyclic_and_sub_polynomial<
        Torus, params::opt, params::degree / params::opt>(
        accumulator_body, accumulator_body_rotated, a_hat);

    synchronize_threads_in_block();

    // Perform a rounding to increase the accuracy of the
    // bootstrapped ciphertext
    round_to_closest_multiple_inplace<Torus, params::opt,
        params::degree / params::opt>(
        accumulator_mask_rotated, base_log, l_gadget);

    round_to_closest_multiple_inplace<Torus, params::opt,
        params::degree / params::opt>(
        accumulator_body_rotated, base_log, l_gadget);
    // Initialize the polynomial multiplication via FFT arrays
    // The polynomial multiplications happens at the block level
    // and each thread handles two or more coefficients
    int pos = threadIdx.x;
    for (int j = 0; j < params::opt / 2; j++) {
      mask_res_fft[pos].x = 0;
      mask_res_fft[pos].y = 0;
      body_res_fft[pos].x = 0;
      body_res_fft[pos].y = 0;
      pos += params::degree / params::opt;
    }

    // Now that the rotation is done, decompose the resulting polynomial
    // coefficients so as to multiply each decomposed level with the
    // corresponding part of the bootstrapping key
    // TODO (Agnes) explain why we do that for the mask and body separately
    for (int decomp_level = 0; decomp_level < l_gadget; decomp_level++) {

      gadget.decompose_one_level(accumulator_mask_decomposed,
                                 accumulator_mask_rotated, decomp_level);

      gadget.decompose_one_level(accumulator_body_decomposed,
                                 accumulator_body_rotated, decomp_level);

      synchronize_threads_in_block();

      // First, perform the polynomial multiplication for the mask

      // Reduce the size of the FFT to be performed by storing
      // the real-valued polynomial into a complex polynomial
      real_to_complex_compressed<int16_t, params>(accumulator_mask_decomposed,
                                                  accumulator_fft);

      synchronize_threads_in_block();
      // Switch to the FFT space
      NSMFFT_direct<HalfDegree<params>>(accumulator_fft);
      synchronize_threads_in_block();

      correction_direct_fft_inplace<params>(accumulator_fft);

      // Get the bootstrapping key piece necessary for the multiplication
      // It is already in the Fourier domain
      // TODO (Agnes) Explain why for the mask polynomial multiplication
      // we need the bsk_body_slice and vice versa
      auto bsk_mask_slice = PolynomialFourier<double2, params>(
          get_ith_mask_kth_block(
              bootstrapping_key, iteration, 0, decomp_level,
              polynomial_size, 1, l_gadget));
      auto bsk_body_slice = PolynomialFourier<double2, params>(
          get_ith_body_kth_block(
              bootstrapping_key, iteration, 0, decomp_level,
              polynomial_size, 1, l_gadget));

      synchronize_threads_in_block();

      // Perform the coefficient-wise product with the two pieces of
      // bootstrapping key TODO (Agnes) why two pieces?
      polynomial_product_accumulate_in_fourier_domain(
          mask_res_fft, accumulator_fft, bsk_mask_slice);
      polynomial_product_accumulate_in_fourier_domain(
          body_res_fft, accumulator_fft, bsk_body_slice);

      synchronize_threads_in_block();

      // Now handle the polynomial multiplication for the body
      // in the same way
      real_to_complex_compressed<int16_t, params>(accumulator_body_decomposed,
                                                  accumulator_fft);
      synchronize_threads_in_block();

      NSMFFT_direct<HalfDegree<params>>(accumulator_fft);
      synchronize_threads_in_block();

      correction_direct_fft_inplace<params>(accumulator_fft);

      auto bsk_mask_slice_2 = PolynomialFourier<double2, params>(
          get_ith_mask_kth_block(bootstrapping_key, iteration, 1, decomp_level,
                                 polynomial_size, 1, l_gadget));
      auto bsk_body_slice_2 = PolynomialFourier<double2, params>(
          get_ith_body_kth_block(bootstrapping_key, iteration, 1, decomp_level,
                                     polynomial_size, 1, l_gadget));

      synchronize_threads_in_block();

      polynomial_product_accumulate_in_fourier_domain(
          mask_res_fft, accumulator_fft, bsk_mask_slice_2);
      polynomial_product_accumulate_in_fourier_domain(
          body_res_fft, accumulator_fft, bsk_body_slice_2);
    }

    // Come back to the coefficient representation
    if constexpr (SMD == FULLSM || SMD == NOSM) {
      synchronize_threads_in_block();

      correction_inverse_fft_inplace<params>(mask_res_fft);
      correction_inverse_fft_inplace<params>(body_res_fft);
      synchronize_threads_in_block();

      NSMFFT_inverse<HalfDegree<params>>(mask_res_fft);
      NSMFFT_inverse<HalfDegree<params>>(body_res_fft);

      synchronize_threads_in_block();

      add_to_torus<Torus, params>(mask_res_fft, accumulator_mask);
      add_to_torus<Torus, params>(body_res_fft, accumulator_body);
      synchronize_threads_in_block();
    } else {
      int tid = threadIdx.x;
#pragma unroll
      for (int i = 0; i < params::opt / 2; i++) {
        accumulator_fft[tid] = mask_res_fft[tid];
        tid = tid + params::degree / params::opt;
      }
      synchronize_threads_in_block();

      correction_inverse_fft_inplace<params>(accumulator_fft);
      synchronize_threads_in_block();

      NSMFFT_inverse<HalfDegree<params>>(accumulator_fft);
      synchronize_threads_in_block();

      add_to_torus<Torus, params>(accumulator_fft, accumulator_mask);
      synchronize_threads_in_block();

      tid = threadIdx.x;
#pragma unroll
      for (int i = 0; i < params::opt / 2; i++) {
        accumulator_fft[tid] = body_res_fft[tid];
        tid = tid + params::degree / params::opt;
      }
      synchronize_threads_in_block();

      correction_inverse_fft_inplace<params>(accumulator_fft);
      synchronize_threads_in_block();

      NSMFFT_inverse<HalfDegree<params>>(accumulator_fft);
      synchronize_threads_in_block();

      add_to_torus<Torus, params>(accumulator_fft, accumulator_body);
      synchronize_threads_in_block();
    }
  }

  auto block_lwe_out = &lwe_out[blockIdx.x * (polynomial_size + 1)];

  // The blind rotation for this block is over
  // Now we can perform the sample extraction: for the body it's just
  // the resulting constant coefficient of the accumulator
  // For the mask it's more complicated TODO (Agnes) explain why
  sample_extract_mask<Torus, params>(block_lwe_out, accumulator_mask);
  sample_extract_body<Torus, params>(block_lwe_out, accumulator_body);
}

template <typename Torus, class params>
__host__ void host_bootstrap_amortized(
    void *v_stream,
    Torus *lwe_out,
    Torus *lut_vector,
    uint32_t *lut_vector_indexes,
    Torus *lwe_in,
    double2 *bootstrapping_key,
    uint32_t input_lwe_dimension,
    uint32_t polynomial_size,
    uint32_t base_log,
    uint32_t l_gadget,
    uint32_t input_lwe_ciphertext_count,
    uint32_t num_lut_vectors,
    uint32_t lwe_idx,
    uint32_t max_shared_memory) {

  int SM_FULL = sizeof(Torus) * polynomial_size +   // accumulator mask
                sizeof(Torus) * polynomial_size +   // accumulator body
                sizeof(Torus) * polynomial_size +   // accumulator mask rotated
                sizeof(Torus) * polynomial_size +   // accumulator body rotated
                sizeof(int16_t) * polynomial_size + // accumulator_dec mask
                sizeof(int16_t) * polynomial_size + // accumulator_dec_body
                sizeof(double2) * polynomial_size / 2 + // accumulator fft mask
                sizeof(double2) * polynomial_size / 2 + // accumulator fft body
                sizeof(double2) * polynomial_size / 2;  // calculate buffer fft

  int SM_PART = sizeof(double2) * polynomial_size / 2; // calculate buffer fft

  int DM_PART = SM_FULL - SM_PART;

  int DM_FULL = SM_FULL;

  auto stream = static_cast<cudaStream_t *>(v_stream);

  char *d_mem;

  // Create a 1-dimensional grid of threads
  // where each block handles 1 sample and each thread
  // handles opt polynomial coefficients
  // (actually opt/2 coefficients since we compress the real polynomial into a
  // complex)
  // TODO (Agnes) Polynomial size / params::opt should be equal to 256 or 512
  //   probably, maybe 1024 would be too big?
  //   Or would it actually be good in our case to have the largest possible
  //   number of threads per block since anyway few blocks will run
  //   concurrently?
  dim3 grid(input_lwe_ciphertext_count, 1, 1);
  dim3 thds(polynomial_size / params::opt, 1, 1);

  // Launch the kernel using polynomial_size/opt threads
  // where each thread computes opt polynomial coefficients
  // Depending on the required amount of shared memory, choose
  // from one of three templates (no use, partial use or full use
  // of shared memory)
  if (max_shared_memory < SM_PART) {
    checkCudaErrors(cudaMalloc((void **)&d_mem, DM_FULL * input_lwe_ciphertext_count));
    device_bootstrap_amortized<Torus, params, NOSM>
    <<<grid, thds, 0, *stream>>>(
        lwe_out, lut_vector, lut_vector_indexes, lwe_in,
        bootstrapping_key, d_mem,
        input_lwe_dimension, polynomial_size,
        base_log, l_gadget, lwe_idx, DM_FULL);
  } else if (max_shared_memory < SM_FULL) {
    cudaFuncSetAttribute(device_bootstrap_amortized<Torus, params, PARTIALSM>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SM_PART);
    cudaFuncSetCacheConfig(
        device_bootstrap_amortized<Torus, params, PARTIALSM>,
        cudaFuncCachePreferShared);
    checkCudaErrors(cudaMalloc((void **)&d_mem, DM_PART * input_lwe_ciphertext_count));
    device_bootstrap_amortized<Torus, params, PARTIALSM>
    <<<grid, thds, SM_PART, *stream>>>(
        lwe_out, lut_vector, lut_vector_indexes,
        lwe_in, bootstrapping_key,
        d_mem, input_lwe_dimension, polynomial_size,
        base_log, l_gadget, lwe_idx,
        DM_PART);
  } else {
    // For devices with compute capability 7.x a single thread block can
    // address the full capacity of shared memory. Shared memory on the
    // device then has to be allocated dynamically.
    // For lower compute capabilities, this call
    // just does nothing and the amount of shared memory used is 48 KB
    checkCudaErrors(cudaFuncSetAttribute(
        device_bootstrap_amortized<Torus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SM_FULL));
    // TODO (Agnes): is this necessary?
    checkCudaErrors(cudaFuncSetCacheConfig(
        device_bootstrap_amortized<Torus, params, FULLSM>,
        cudaFuncCachePreferShared));
    checkCudaErrors(cudaMalloc((void **)&d_mem, 0));

    device_bootstrap_amortized<Torus, params, FULLSM>
    <<<grid, thds, SM_FULL, *stream>>>(
        lwe_out, lut_vector, lut_vector_indexes,
        lwe_in, bootstrapping_key,
        d_mem, input_lwe_dimension, polynomial_size,
        base_log, l_gadget, lwe_idx,
        0);
  }
  // Synchronize the streams before copying the result to lwe_out at the right
  // place
  cudaStreamSynchronize(*stream);
  cudaFree(d_mem);

}

template <typename Torus, class params>
int cuda_get_pbs_per_gpu(int polynomial_size) {

  int blocks_per_sm = 0;
  int num_threads = polynomial_size / params::opt;
  cudaGetDeviceCount(0);
  cudaDeviceProp device_properties;
  // FIXME: here we assume every device has same properties
  cudaGetDeviceProperties(&device_properties, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, device_bootstrap_amortized<Torus, params>,
          num_threads, 0);

  return device_properties.multiProcessorCount * blocks_per_sm;
}

#endif // CNCRT_PBS_H
