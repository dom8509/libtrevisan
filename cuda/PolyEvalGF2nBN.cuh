#ifndef __MONTGOMERY_CUH__
#define __MONTGOMERY_CUH__

#include "CudaTypes.h"

#ifdef CUDA_SANITY_CHECKS
#include <iostream>
#include <fstream>
#endif

#ifdef INCLUDE_FROM_CUDA_FILE
#define CUDA_CALLABLE_MEMBER __host__
#else
#define CUDA_CALLABLE_MEMBER
#endif

/*
	Public Functions
*/
class GF2nPolyBN 
{
public:
	CUDA_CALLABLE_MEMBER GF2nPolyBN(sfixn* coeffs, sfixn* x, sfixn num_x, sfixn size_field, sfixn deg_poly, sfixn* irred_poly, sfixn* mask);
	CUDA_CALLABLE_MEMBER ~GF2nPolyBN();

	CUDA_CALLABLE_MEMBER void evaluate(sfixn i);
	CUDA_CALLABLE_MEMBER void getResults(sfixn *result);

private:
	CUDA_CALLABLE_MEMBER void loadPoroperties();
	void initBarrier(sfixn num_blocks);

private:
	sfixn* m_dx;
	sfixn* m_dCoeffs;
	sfixn* m_dIrred_poly;
	sfixn* m_dMask;
	sfixn* m_dTmp1;
	sfixn* m_dTmp2;
	sfixn* m_dTmp_long;
	sfixn* m_dTmp_Result;

	sfixn* m_dbarrier;

	sfixn m_num_x;
	sfixn m_size_field;
	sfixn m_num_coeffs;
	sfixn m_num_chunks;
	sfixn m_width_binary_tree;
	sfixn m_bytes_for_chunks;

	sfixn m_hMaxThreadsPerBlock;
	sfixn m_hSharedMemPerBlock;

	sfixn* m_hTmp_Result;

#ifdef CUDA_SANITY_CHECKS
	std::ofstream *m_result_file;
#endif
};

#ifdef INCLUDE_FROM_CUDA_FILE
/*
	Host Functions
*/
__host__ sfixn getCeilToPotOf2n(sfixn value);
__host__ void padWithZeros(sfixn* data_old, sfixn size_old, sfixn num_chunks, sfixn* data_new, sfixn size_new);
__host__ sfixn getNumberBlocksForSharedMem(sfixn sharedMemSize );

/*
	Kernel Functions
*/
__global__ void cudaPrefProdReduce(sfixn num_chunks, sfixn* irred_poly, sfixn* mask, sfixn length_exp_tree, sfixn* tmp, sfixn* res);
__global__ void cudaPrefProdDownSweep(sfixn num_chunks, sfixn* irred_poly, sfixn* mask, sfixn length_exp_tree, sfixn* tmp1, sfixn* tmp2, sfixn* res);
__global__ void cudaMontgMulBNKernel(sfixn* values, sfixn* x_preCalc, sfixn num_values, sfixn num_chunks, sfixn* irred_poly, sfixn* mask, sfixn* tmp, sfixn* res);
__global__ void cudaBitSumBNKernel(sfixn* values, sfixn length_values, sfixn n);
__global__ void cudaCopyBNKernel(sfixn* a, sfixn num_chunks_a, sfixn* b, sfixn num_chunks_b);
__global__ void cudaExpandVecBNKernel(sfixn* value, sfixn blocks_per_value, sfixn* value_vec, sfixn blocks_value_vec);
__global__ void cudaSet0Kernel(sfixn* x, sfixn length);
__global__ void cudaSet1Kernel(sfixn* x, sfixn length);
__global__ void cudaBitAddBNKernel(sfixn* a, sfixn* b, sfixn num_chunks);
__global__ void cudaPrintbincharpadKernel(sfixn* ca, unsigned int n);
__global__ void cudainitBarriersKernel(sfixn* barriers, sfixn num_barriers, sfixn num_blocks);

/*
	Device Functions
*/
__device__ void cudaMontgMulBN(sfixn* a, sfixn* b, sfixn num_chunks, sfixn* irred_poly, sfixn* mask, sfixn* tmp, sfixn* res);
__device__ void cudaReducePolyBN(sfixn* value, sfixn num_chunks, sfixn* irred_poly, sfixn* mask);
__device__ bool cudaBitCheckBN(sfixn* x, sfixn num_chunks);
__device__ void cudaBitShiftLeft1BN(sfixn* a, sfixn num_chunks);
__device__ void cudaPrintbincharpad(sfixn* ca, unsigned int n);

__host__ __device__ sfixn getNumberChunks(sfixn length);
__host__ __device__ sfixn isbitset(sfixn val, sfixn bitnum);

__global__ void testMontgMult(sfixn* a, sfixn* b, sfixn num_chunks,sfixn* irred_poly, sfixn* mask,sfixn* tmp,sfixn* res);
__global__ void testShift(sfixn* x, sfixn num_chunks);

#endif

#endif