#ifndef __MONTGOMERY_CUH__
#define __MONTGOMERY_CUH__

#include "CudaTypes.h"

/*
	Public Functions
*/
void evaluateGF2nPolyBN(sfixn* coeffs, sfixn* x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask, sfixn* result);


#ifdef INCLUDE_FROM_CUDA_FILE
/*
	Host Functions
*/
__host__ void loadPoroperties();
__host__ sfixn getCeilToPotOf2n(sfixn value);
__host__ void padWithZeros(sfixn* data_old, sfixn size_old, sfixn num_chunks, sfixn* data_new, sfixn size_new);
__host__ sfixn getNumberBlocksForSharedMem(sfixn sharedMemSize );

/*
	Kernel Functions
*/
__global__ void cudaCreateExpTreeBNKernel(sfixn* x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask, sfixn length_exp_tree, sfixn* res);
__global__ void cudaMontgMulBNKernel(sfixn* values, sfixn* x_preCalc, sfixn num_values, sfixn num_chunks, sfixn* irred_poly, sfixn* res);
__global__ void cudaBitSumBNKernel(sfixn* values, sfixn length_values, sfixn n);
__global__ void cudaCopyBNKernel(sfixn* a, sfixn num_chunks_a, sfixn* b, sfixn num_chunks_b);
__global__ void cudaExpandVecBNKernel(sfixn* value, sfixn blocks_per_value, sfixn* value_vec, sfixn blocks_value_vec);
__global__ void cudaSet0Kernel(sfixn* x, sfixn length);
__global__ void cudaBitAddBNKernel(sfixn* a, sfixn* b, sfixn num_chunks);

/*
	Device Functions
*/
__device__ void cudaMontgMulBN(sfixn* a, sfixn* b, sfixn num_chunks, sfixn* irred_poly, sfixn* res);
__device__ void cudaReducePolyBN(sfixn* value, sfixn num_chunks, sfixn* irred_poly, sfixn pos_msb);
__device__ bool cudaBitCheckBN(sfixn* x, sfixn num_chunks);
__device__ void cudaBitShiftLeft1BN(sfixn* a, sfixn num_chunks);

__host__ __device__ sfixn getNumberChunks(sfixn length);
__host__ __device__ sfixn isbitset(sfixn val, sfixn bitnum);

#endif

#endif