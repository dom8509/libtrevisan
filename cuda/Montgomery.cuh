#ifndef __MONTGOMERY_CUH__
#define __MONTGOMERY_CUH__

#include "libtrevisancuda.h"

/*
	Host Functions
*/
void evaluateGF2nPolyBN(sfixn* coeffs, sfixn* x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask, sfixn* result);
__host__ sfixn getCeilToPotOf2n(sfixn value);
__host__ void padWithZeros(sfixn* data_old, sfixn size_old, sfixn size_field, sfixn* data_new, sfixn size_new);

/*
	Kernel Functions
*/
__global__ void cudaCreateExpTreeBNKernel(sfixn* x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask, sfixn length_exp_tree, sfixn* res);
__global__ void cudaMontgMulBNKernel(sfixn* values, sfixn* x_preCalc, sfixn field_size, sfixn* irred_poly, sfixn length_irred_poly, sfixn* irred_poly_mask, sfixn* res);
__global__ void cudaBitSumBNKernel(sfixn* values, sfixn length_values, sfixn n);
__global__ void cudaCopyBNKernel(sfixn* a, sfixn num_chunks_a, sfixn* b, sfixn num_chunks_b);
__global__ void cudaExpandVecBNKernel(sfixn* value, sfixn blocks_per_value, sfixn* value_vec, sfixn blocks_value_vec) ;
__global__ void cudaSet0Kernel(sfixn* x, sfixn length);
__global__ void cudaBitAddBNKernel(sfixn* a, sfixn* b, sfixn num_chunks);
// __global__ void cudaMontgMulBN
// __global__ void cudaCreateExpTreeBN
// __global__ void cudaBitShiftRightBN
// __global__ void cudaCopyBN

/*
	Device Functions
*/
// __device__ void cudaMontgMulBN
// __device__ void cudaBitAddBN
// __device__ void cudaBitAndBN
// __device__ bool cudaBitCheckBN
__device__ void cudaMontgMulBN(sfixn* a, sfixn* b, sfixn size_field, sfixn* irred_poly, sfixn* irred_poly_mask, sfixn* tmp_a, sfixn* tmp_b, sfixn* res);
__host__ __device__ sfixn getNumberChunks(sfixn length);
__device__ void cudaReducePolyBN(sfixn* coeffs, sfixn num_chunks_coeffs, sfixn* irred_poly, sfixn* irred_poly_mask, sfixn* tmp);
__device__ bool cudaBitCheckBN(sfixn* x, sfixn num_chunks);

#endif