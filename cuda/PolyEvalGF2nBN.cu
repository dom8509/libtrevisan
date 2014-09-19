#define INCLUDE_FROM_CUDA_FILE
#include "PolyEvalGF2nBN.cuh"
#undef INCLUDE_FROM_CUDA_FILE

#ifdef CUDA_SANITY_CHECKS
#include "../utils.hpp"
#endif

#include "CudaUtils.h"

////////////////////////////////////////////////////////////////////////////////
/*
	Some constant memory variables
*/
////////////////////////////////////////////////////////////////////////////////
// Device side
__constant__ sfixn dMaxThreadsPerBlock;
__constant__ sfixn dSharedMemPerBlock;

__global__ void testMontgMult(sfixn* a, sfixn* b, sfixn num_chunks,sfixn* irred_poly, sfixn* mask,sfixn* tmp,sfixn* res) {

	cudaMontgMulBN(a, b, num_chunks, irred_poly, mask, tmp, res);
	//cudaMontgMulBN(NULL, NULL, 0, NULL, NULL, NULL, NULL);
	//cudaBitCheckBN(a, num_chunks);
}

__global__ void testShift(sfixn* x, sfixn num_chunks) {
	cudaBitShiftLeft1BN(x, num_chunks);
}


sfixn *dx, *dCoeffs, *dIrred_poly, *dMask, *dTmp1, *dTmp2, *dTmp_long, *dTmp_Result;

////////////////////////////////////////////////////////////////////////////////
/*
	Host Functions
*/
////////////////////////////////////////////////////////////////////////////////

CUDA_CALLABLE_MEMBER GF2nPolyBN::GF2nPolyBN(
	sfixn* coeffs, 
	sfixn* x,
	sfixn num_x,
	sfixn size_field,
	sfixn deg_poly, 
	sfixn* irred_poly,
	sfixn* mask ) {

	m_num_x = num_x;
	m_size_field = size_field;

	// How many coefficients does our polynomial have
	m_num_coeffs = deg_poly + 1;

	// Calculate the number of chunks of the data elements
	m_num_chunks = getNumberChunks(m_size_field + 1);

	// Calcualte the amount of leaves needed for used binary tree method
	m_width_binary_tree = getCeilToPotOf2n(m_num_coeffs);

	// How many bytes do we need for one field element
	m_bytes_for_chunks = m_num_chunks * SIZE_CHUNK / SIZE_BYTE;

#ifdef CUDA_SANITY_CHECKS
	std::cout << "Creating Result file..." << std::endl;

	remove("rsh_test_results");

	m_result_file = new std::ofstream();
	m_result_file->open("rsh_test_results");

	// write input parameters
	(*m_result_file) << "field_size=";
	printbinToFile(&m_size_field, 1, 1, m_result_file);
	(*m_result_file) << "\n";
	(*m_result_file) << "num_coeffs=";
	printbinToFile(&m_num_coeffs, 1, 1, m_result_file);
	(*m_result_file) << "\n";
	(*m_result_file) << "coeffs=";
	printbinToFile(coeffs, m_num_chunks, m_num_coeffs, m_result_file);
	(*m_result_file) << "\n";
	(*m_result_file) << "x=";
	printbinToFile(x, m_num_chunks, 1, m_result_file);
	(*m_result_file) << "\n";
	(*m_result_file) << "irred_poly=";
	printbinToFile(irred_poly, m_num_chunks, 1, m_result_file);
	(*m_result_file) << "\n";
	(*m_result_file) << "mask=";
	printbinToFile(mask, m_num_chunks, 1, m_result_file);
	(*m_result_file) << "\n";
#endif

	loadPoroperties();

	// Allocate global memory
	CudaSafeCall(cudaMalloc((sfixn**)&m_dx, m_num_x * m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dCoeffs, m_width_binary_tree * m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dIrred_poly, m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dMask, m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dTmp1, m_width_binary_tree / 2 * m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dTmp2, m_width_binary_tree / 2 * m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dTmp_long, m_width_binary_tree * m_bytes_for_chunks));
	CudaSafeCall(cudaMalloc((sfixn**)&m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks));

	sfixn hCoeffs[m_width_binary_tree * m_bytes_for_chunks];
	padWithZeros(coeffs, m_num_coeffs, m_num_chunks, hCoeffs, m_width_binary_tree);

	// Copy host data to device
	CudaSafeCall(cudaMemcpy(m_dx, x, m_num_x * m_bytes_for_chunks, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(m_dCoeffs, hCoeffs, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(m_dIrred_poly, irred_poly, m_bytes_for_chunks, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(m_dMask, mask, m_bytes_for_chunks, cudaMemcpyHostToDevice));

	m_hTmp_Result = new sfixn[m_width_binary_tree * m_bytes_for_chunks];

}

CUDA_CALLABLE_MEMBER GF2nPolyBN::~GF2nPolyBN() {

	// Free all allocated device memory
	CudaSafeCall(cudaFree(m_dx));
	CudaSafeCall(cudaFree(m_dCoeffs));
	CudaSafeCall(cudaFree(m_dIrred_poly));
	CudaSafeCall(cudaFree(m_dMask));
	CudaSafeCall(cudaFree(m_dTmp1));
	CudaSafeCall(cudaFree(m_dTmp2));
	CudaSafeCall(cudaFree(m_dTmp_long));
	CudaSafeCall(cudaFree(m_dTmp_Result));

	delete m_hTmp_Result;

#ifdef CUDA_SANITY_CHECKS
	std::cout << "finished" << std::endl;
	m_result_file->close();
	delete m_result_file;
#endif

}

////////////////////////////////////////////////////////////////////////////////
//
//	A polynomial evaluation over GF(2n) with BN Coefficients
//
//	Evaluates a polynomial with BN coefficients in GF(2n)
// 	at point x. Note that the size of the field elements (coeffs, x, result)
//	must be big enough to store an additional carry bit (size = size_field + 1)!
//
//	Input:
//		coeffs 			- the coefficients of the polynomial		-> size = (k + 1) * (n + 1)
//		x 				- point where the polynomial is evaluated	-> size = n + 1
//		size_field  	- number of bits of the field elements		-> n
//		deg_poly 		- degree of the evaluated polynomial 		-> k
//		irred_poly 		- the irreducible polynomial of the GF(2n) 	-> size = n + 1
// 		mask 			- mask of the field 						-> size = n + 1
//		result			- the result 								-> size = n + 1
//
////////////////////////////////////////////////////////////////////////////////
CUDA_CALLABLE_MEMBER void GF2nPolyBN::evaluate(sfixn i) {

	sfixn num_threads, num_blocks;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	num_threads = min(m_num_chunks*m_width_binary_tree, m_hMaxThreadsPerBlock);
	num_blocks = ceil((double)m_num_chunks*m_width_binary_tree/m_hMaxThreadsPerBlock);
	printf("Starting expand step...\n");
	cudaExpandVecBNKernel<<<num_blocks, num_threads>>>(&m_dx[i * m_num_chunks], m_num_chunks, m_dTmp_Result, m_num_chunks*m_width_binary_tree);
	cudaDeviceSynchronize();
#ifdef CUDA_SANITY_CHECKS
	CudaSafeCall(cudaMemcpy(m_hTmp_Result, m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyDeviceToHost));
	(*m_result_file) << "resultExpandStep=";
	printbinToFile(m_hTmp_Result, m_num_chunks, m_width_binary_tree, m_result_file);
	(*m_result_file) << "\n";
#endif

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create all exponentiation from x^0 to x^deg_poly and store it in res 
	num_threads = min(m_width_binary_tree/2, m_hMaxThreadsPerBlock);
	num_blocks = ceil((double)m_width_binary_tree/2/m_hMaxThreadsPerBlock);
	// Calculate reduce step
	printf("Starting reduce step...\n");
	cudaPrefProdReduce<<<num_blocks, num_threads>>>(m_num_chunks, m_dIrred_poly, m_dMask, m_width_binary_tree, m_dTmp1, m_dTmp_Result);
	cudaDeviceSynchronize();
#ifdef CUDA_SANITY_CHECKS
	CudaSafeCall(cudaMemcpy(m_hTmp_Result, m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyDeviceToHost));
	(*m_result_file) << "resultReduceStep=";
	printbinToFile(m_hTmp_Result, m_num_chunks, m_width_binary_tree, m_result_file);
	(*m_result_file) << "\n";
#endif

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// calculate down sweep step
	printf("Starting down swep step...\n");
	cudaPrefProdDownSweep<<<num_blocks, num_threads>>>(m_num_chunks, m_dIrred_poly, m_dMask, m_width_binary_tree, m_dTmp1, m_dTmp2, m_dTmp_Result);
	cudaDeviceSynchronize();
#ifdef CUDA_SANITY_CHECKS
	CudaSafeCall(cudaMemcpy(m_hTmp_Result, m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyDeviceToHost));
	(*m_result_file) << "resultDownSweepStep=";
	printbinToFile(m_hTmp_Result, m_num_chunks, m_width_binary_tree, m_result_file);
	(*m_result_file) << "\n";
#endif

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Multiply each coefficient with its related exponentiation of x 
	// (coeff[0]*x^0, ..., coeff[deg_poly]*x^deg_poly) and store it in res
	num_threads = min(m_width_binary_tree, m_hMaxThreadsPerBlock);
	num_blocks = ceil((double)m_width_binary_tree/m_hMaxThreadsPerBlock);

	printf("Starting prod step...\n");
	cudaMontgMulBNKernel<<<num_blocks, num_threads>>>(m_dCoeffs, m_dTmp_Result, m_width_binary_tree, m_num_chunks, m_dIrred_poly, m_dMask, m_dTmp_long, m_dTmp_Result);
	cudaDeviceSynchronize();
#ifdef CUDA_SANITY_CHECKS
	CudaSafeCall(cudaMemcpy(m_hTmp_Result, m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyDeviceToHost));
	(*m_result_file) << "resultProdStep=";
	printbinToFile(m_hTmp_Result, m_num_chunks, m_width_binary_tree, m_result_file);
	(*m_result_file) << "\n";
#endif

	// Add all summands of the polynom up to the result
	num_threads = min(m_width_binary_tree/2, m_hMaxThreadsPerBlock);
	num_blocks = ceil((double)m_width_binary_tree/2/m_hMaxThreadsPerBlock);

	printf("Starting sum step...\n");
	cudaBitSumBNKernel<<<num_blocks, num_threads>>>(m_dTmp_Result, m_num_chunks, m_width_binary_tree);
	cudaDeviceSynchronize();

#ifdef CUDA_SANITY_CHECKS
	CudaSafeCall(cudaMemcpy(m_hTmp_Result, m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyDeviceToHost));
	(*m_result_file) << "resultSumStep=";
	printbinToFile(m_hTmp_Result, m_num_chunks, 1, m_result_file);
	(*m_result_file) << "\n";
#endif	

}

CUDA_CALLABLE_MEMBER void GF2nPolyBN::getResults(sfixn *result) {

	// Read the result from device
	CudaSafeCall(cudaMemcpy(m_hTmp_Result, m_dTmp_Result, m_width_binary_tree * m_bytes_for_chunks, cudaMemcpyDeviceToHost));

	// Create result
	for( sfixn i=0; i<m_num_chunks; ++i ) {
		result[i] = m_hTmp_Result[i];
	}

}

////////////////////////////////////////////////////////////////////////////////
//
//	Load some cuda properties to constant memory
//
////////////////////////////////////////////////////////////////////////////////
CUDA_CALLABLE_MEMBER void GF2nPolyBN::loadPoroperties() {

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	CudaSafeCall(cudaMemcpyToSymbol(dMaxThreadsPerBlock, (const char *)&deviceProp.maxThreadsPerBlock, sizeof(sfixn), 0, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpyToSymbol(dSharedMemPerBlock, &deviceProp.sharedMemPerBlock, sizeof(sfixn)));

	m_hMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	m_hSharedMemPerBlock = deviceProp.sharedMemPerBlock;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds leading zero blocks to the array so that its new size matches size_new
//
////////////////////////////////////////////////////////////////////////////////
__host__ void padWithZeros( 
	sfixn* data_old, 
	sfixn size_old, 
	sfixn num_chunks, 
	sfixn* data_new, 
	sfixn size_new ) {

	for( sfixn i=size_new*num_chunks-1; i>=0; --i ) {
		if( i >= size_new - size_old ) {
			data_new[i] = data_old[i - (size_new - size_old)];
		} else {
			data_new[i] = 0;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Returns the next higher pot of 2 of the passed value
//
////////////////////////////////////////////////////////////////////////////////
__host__ sfixn getCeilToPotOf2n( sfixn value ) {
	
	sfixn res = 0;
	
	if( value > 0 ) {
		res = 1;
		while(res < value) {
			res<<=1;
		}
	}

	return res;
}

__host__ sfixn getNumberBlocksForSharedMem( sfixn sharedMemSize ) {

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
/*
	Kernel Functions
*/
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//	Callculates all exponentiation of x to the power of n
//
//	Input:
//		x 				- point where the polynomial is evaluated
//		length_x	 	- number of bits of x
//		irred_poly 		- the irreducible polynomial of the GF(2n)
//		y				- the result
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaPrefProdReduce(
	sfixn num_chunks,
	sfixn* irred_poly,
	sfixn* mask, 
	sfixn length_exp_tree,
	sfixn* tmp,
	sfixn* res) {

    sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if( thid < length_exp_tree/2 ) {

		sfixn *local_tmp = &tmp[thid * num_chunks];			

	    sfixn offset = 1;

	    int i = 0;

	 	for( sfixn d=length_exp_tree>>1; d>0; d>>=1 ) { //build sum in place up the tree 
	 		__syncthreads();

	 		if( thid < d ) {
	 			sfixn ai = (offset * (2*thid+1) - offset) * num_chunks;
	 			sfixn bi = (offset * (2*thid+2) - offset) * num_chunks;

				cudaMontgMulBN(
					&res[ai], 
					&res[bi],
					num_chunks,
					irred_poly,
					mask,
					local_tmp,
					&res[ai]);

				__syncthreads();
	 		}
	 		__syncthreads();

	 		offset <<= 1;

	 		++i;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Callculates all exponentiation of x to the power of n
//
//	Input:
//		x 				- point where the polynomial is evaluated
//		length_x	 	- number of bits of x
//		irred_poly 		- the irreducible polynomial of the GF(2n)
//		y				- the result
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaPrefProdDownSweep(
	sfixn num_chunks,
	sfixn* irred_poly,
	sfixn* mask, 
	sfixn length_exp_tree,
	sfixn* tmp1,
	sfixn* tmp2,
	sfixn* res) {

    sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	sfixn num_threads, num_blocks;

    if( thid < length_exp_tree/2 ) {

		sfixn *local_tmp1 = &tmp1[thid * num_chunks];
		sfixn *local_tmp2 = &tmp2[thid * num_chunks];

		num_threads = min(num_chunks, dMaxThreadsPerBlock);
		num_blocks = ceil((double)num_chunks/dMaxThreadsPerBlock);
		
		if( thid == 0 ) {
			cudaSet0Kernel<<<num_blocks, num_threads>>>(&local_tmp1[0], num_chunks);
			cudaSet1Kernel<<<num_blocks, num_threads>>>(&res[0], num_chunks);
		}

		sfixn offset = length_exp_tree;

		sfixn i = 0;

		for( sfixn d=1; d<length_exp_tree; d*=2 ) { //traverse down tree & build scan 

		    offset >>= 1;
		    __syncthreads();

			if( thid < d ) {
				sfixn ai = (offset * (2*thid+1) - offset) * num_chunks;
		 		sfixn bi = (offset * (2*thid+2) - offset) * num_chunks;

		 		__syncthreads();

				cudaCopyBNKernel<<<num_blocks, num_threads>>>(&res[bi], num_chunks, local_tmp1, num_chunks);
				__syncthreads();
				if( thid == 0 ) {
					//printf("local_tmp1 in iteration i: "); cudaPrintbincharpad(local_tmp1, num_chunks);
				}
				cudaCopyBNKernel<<<num_blocks, num_threads>>>(&res[ai], num_chunks, &res[bi], num_chunks);
				__syncthreads();

				cudaMontgMulBN(
					&res[ai], 
					local_tmp1,
					num_chunks,
					irred_poly,
					mask,
					local_tmp2,
					&res[ai]);
				__syncthreads();

				++i;
		    }
		}

		__syncthreads();
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Shifts a n bits to the right
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitShiftRightBNKernel(sfixn* a, sfixn length_a, sfixn n, sfixn* c) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(thid < getNumberChunks(length_a)) {

		sfixn min_range = thid * SIZE_CHUNK; //<- Block size
		//sfixn max_range = min_range + SIZE_CHUNK - 1;

		sfixn min_src_range = min_range - n;
		sfixn max_src_range = min_src_range + SIZE_CHUNK - 1;

		if(max_src_range < 0) {
			c[thid] = 0;
		} else {
			sfixn digits_block_b_shift = (max_src_range + 1) % SIZE_CHUNK;
			if(min_src_range < 0) {
				sfixn mask_b = pow((double)2, (double)SIZE_CHUNK - 1) - pow((double)2, (double)(SIZE_CHUNK-(max_src_range%SIZE_CHUNK)) - 1);
				sfixn src_idx_b = max_src_range/SIZE_CHUNK;
				c[thid] = ((a[src_idx_b]&mask_b) >> (SIZE_CHUNK-(max_src_range%SIZE_CHUNK)));
			} else if(digits_block_b_shift == 0) {
				c[thid] = a[min_src_range/SIZE_CHUNK];
			} else {
				sfixn mask_a = pow((double)2, (double)(SIZE_CHUNK-(min_src_range%SIZE_CHUNK))) - 1;
				sfixn mask_b = (pow((double)2, (double)SIZE_CHUNK) - 1) - (pow((double)2, (double)(SIZE_CHUNK-(max_src_range%SIZE_CHUNK))) - 1);

				sfixn src_idx_a = min_src_range/SIZE_CHUNK;
				sfixn src_idx_b = max_src_range/SIZE_CHUNK;

				c[thid] = ((a[src_idx_a]&mask_a) << (min_src_range%SIZE_CHUNK)) | 
					((a[src_idx_b]&mask_b) >> (SIZE_CHUNK-(max_src_range%SIZE_CHUNK)));

			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Shifts a n bits to the left
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitShiftLeftBNKernel(sfixn* a, sfixn num_chunks, sfixn n, sfixn* c) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks ) {

		sfixn min_range = thid * SIZE_CHUNK; //<- Block size
		//sfixn max_range = min_range + SIZE_CHUNK - 1;

		sfixn min_src_range = min_range + n;
		sfixn max_src_range = min_src_range + SIZE_CHUNK - 1;

		if( min_src_range >= num_chunks * SIZE_CHUNK ) {
			c[thid] = 0;
		} else {
			bool shift_whole_block = ((max_src_range + 1) % SIZE_CHUNK == 0);
			if( max_src_range >= num_chunks * SIZE_CHUNK ) {
				sfixn mask_b = pow((double)2, (double)(SIZE_CHUNK-(min_src_range % SIZE_CHUNK))) - 1;
				sfixn src_idx_b = min_src_range/SIZE_CHUNK;
				c[thid] = ((a[src_idx_b]&mask_b) << (min_src_range%SIZE_CHUNK));
			} else if( shift_whole_block ) {
				c[thid] = a[min_src_range/SIZE_CHUNK];
			} else {
				sfixn mask_a = pow((double)2, (double)(SIZE_CHUNK-(min_src_range%SIZE_CHUNK))) - 1;
				sfixn mask_b = (pow((double)2, (double)SIZE_CHUNK) - 1) - (pow((double)2, (double)(SIZE_CHUNK-((max_src_range+1)%SIZE_CHUNK))) - 1);

				sfixn src_idx_a = min_src_range/SIZE_CHUNK;
				sfixn src_idx_b = max_src_range/SIZE_CHUNK;

				c[thid] = ((a[src_idx_a]&mask_a) << (min_src_range%SIZE_CHUNK)) | 
					((a[src_idx_b]&mask_b) >> (SIZE_CHUNK-((max_src_range+1)%SIZE_CHUNK)));

			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Copies a to b
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaCopyBNKernel( sfixn* a, sfixn num_chunks_a, sfixn* b, sfixn num_chunks_b ) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks_b ) {
		if( thid < num_chunks_a )
			b[thid] = a[thid];
		else
			b[thid] = 0;
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	A binary montgomery multiplication Kernel over big numbers
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaMontgMulBNKernel(
	sfixn* values,
	sfixn* x_preCalc,
	sfixn num_values,
	sfixn num_chunks, 
	sfixn* irred_poly,
	sfixn* mask,
	sfixn* tmp,
	sfixn* res) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// if(thid == 0) {
	// 	printf("values: \n"); 
	// 	for(int i=0; i<num_values; ++i) {
	// 		printf("values[%i]: ", i); cudaPrintbincharpad(&values[i], num_chunks);
	// 	}
	// 	printf("x_preCalc: \n"); 
	// 	for(int i=0; i<num_values; ++i) {
	// 		printf("x_preCalc[%i]: ", i); cudaPrintbincharpad(&x_preCalc[i], num_chunks);
	// 	}
	//}

	cudaMontgMulBN(
		&values[thid*num_chunks], 
		&x_preCalc[thid*num_chunks],
		num_chunks,
		irred_poly,
		mask,
		&tmp[thid*num_chunks],
		&res[thid*num_chunks]);
}

////////////////////////////////////////////////////////////////////////////////
//
//	Sums up all passed values
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitSumBNKernel(
	sfixn* values, 
	sfixn num_chunks,
	sfixn n ) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	int i = 0;

	if( thid < n/2 ) {
		sfixn offset = 1;

		for( sfixn d=n>>1; d>0; d>>=1 ) {
	 		__syncthreads();

	 		if( thid < d ) {
	 			sfixn ai = (offset * (2*thid+1) - offset) * num_chunks;
	 			sfixn bi = (offset * (2*thid+2) - offset) * num_chunks;

	 			// Calculate values_bi = values_ai + values_bi
				sfixn num_threads = min(num_chunks, dMaxThreadsPerBlock);
				sfixn num_blocks = ceil((double)num_chunks/dMaxThreadsPerBlock);
				cudaBitAddBNKernel<<<num_blocks, num_threads>>>(&values[ai], &values[bi], num_chunks);

				__syncthreads();
	 		}
	 		++i;
	 		offset <<= 1;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Creates a vector that contains n instances of the value
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaExpandVecBNKernel(
	sfixn* value,
	sfixn blocks_per_value,
	sfixn* value_vec,
	sfixn blocks_value_vec
	) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < blocks_value_vec ) {
		value_vec[thid] = value[thid % blocks_per_value];
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Sets x to 0
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaSet0Kernel( sfixn* x, sfixn length ) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < length )
		x[thid] = 0;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Sets x to 1
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaSet1Kernel( sfixn* x, sfixn length ) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < length )
		if( thid == length-1 )
			x[thid] = 1;
		else
			x[thid] = 0;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in a
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitAddBNKernel(sfixn* a, sfixn* b, sfixn num_chunks) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
		a[thid] ^= b[thid];
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in c
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitAndBNKernel(sfixn* a, sfixn* b, sfixn* c, sfixn num_chunks) {

	sfixn thid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( thid < num_chunks )
		c[thid] = a[thid] & b[thid];
}

////////////////////////////////////////////////////////////////////////////////
/*
	Device Functions
*/
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//	A binary montgomery multiplication over big numbers
//
////////////////////////////////////////////////////////////////////////////////
__device__ void cudaMontgMulBN(
	sfixn* a, 
	sfixn* b, 
	sfixn num_chunks,
	sfixn* irred_poly, 
	sfixn* mask,
	sfixn* tmp,
	sfixn* res) {

	sfixn num_threads = min(num_chunks, dMaxThreadsPerBlock);
	sfixn num_blocks = ceil((double)num_chunks/dMaxThreadsPerBlock);

	cudaSet0Kernel<<<num_blocks, num_threads>>>(tmp, num_chunks);

	int iteration = 0;

	// if( threadIdx.x == 100 ) {	
	// 	printf("thread 1 started...\n");
	// 	printf("a: "); cudaPrintbincharpad(a, num_chunks);
	// 	printf("b: "); cudaPrintbincharpad(b, num_chunks);
	// 	printf("res: "); cudaPrintbincharpad(res, num_chunks);
	// 	printf("tmp: "); cudaPrintbincharpad(tmp, num_chunks);
	// 	printf("mask: "); cudaPrintbincharpad(mask, num_chunks);
	// 	printf("irred_poly: "); cudaPrintbincharpad(irred_poly, num_chunks);
	// 	printf("dMaxThreadsPerBlock: %i\n", dMaxThreadsPerBlock);
	// 	printf("num_chunks: %i\n", num_chunks); 
	// 	printf("num blocks: %i\n", num_blocks);
	// 	printf("num threads: %i\n", num_threads);
	// }

	bool hit = 0;

	for( sfixn i=num_chunks-1; i>=0; --i ) {

		//__syncthreads();

		for( sfixn j=0; j<SIZE_CHUNK; ++j ) {
		
			//printf("Tmp in iteration %i: ", j); cudaPrintbincharpad(tmp, num_chunks);
			//printf("a in iteration %i: ", j); cudaPrintbincharpad(a, num_chunks);

			//__syncthreads();
			// if(i == num_chunks-1 && j<10) {
			// 	//printf("b[%i]: ", i); cudaPrintbincharpad(&b[i], 1);
			// 	printf("bit: %i\n", j);
			// }
			if( isbitset(b[i], j) ) {
				hit = true;
				//f(i == num_chunks-1)
					//printf("hit\n");
				cudaBitAddBNKernel<<<num_blocks, num_threads>>>(tmp, a, num_chunks);
			} else {
				hit = false;
				//if(i == num_chunks-1 )
					//printf("no hit\n");
			}

			//__syncthreads();
			
			// TODO: There seems to be a bug here if the printf is removed.
			// 		 In this case tmp is shifted after the addition for an unknown reason
			//printf("");
			// if(i == num_chunks-1 && j<10 && hit) {
			// 	printf("a before: ", j); cudaPrintbincharpad(a, num_chunks);
			// }

			cudaBitShiftLeft1BN(a, num_chunks);
			// if(i == num_chunks-1 && j<10 && hit) {
			// 	printf("a after shift: "); cudaPrintbincharpad(a, num_chunks);
			// }
			//__syncthreads();
			cudaReducePolyBN(a, num_chunks, irred_poly, mask);
			//__syncthreads();
						
			//printf("result 2 in iteration %i: ", j); cudaPrintbincharpad(tmp, num_chunks);

			//__syncthreads();
			// if(i == num_chunks-1 && j<10) {
			// 	printf("result: ", j); cudaPrintbincharpad(tmp, num_chunks);
			// 	if(hit) {
			// 		printf("a: ", j); cudaPrintbincharpad(a, num_chunks);
			// 	}	
			// }
		}
	}

	cudaCopyBNKernel<<<num_blocks, num_threads>>>(tmp, num_chunks, res, num_chunks);
}

////////////////////////////////////////////////////////////////////////////////
//
//	Reduce the extended field polynomial
//
////////////////////////////////////////////////////////////////////////////////
__device__ void cudaReducePolyBN(
	sfixn* value,
	sfixn num_chunks,
	sfixn* irred_poly,
	sfixn* mask
	) {

	if( (value[0] & mask[0]) ) {
		sfixn num_threads = min(num_chunks, dMaxThreadsPerBlock);
		sfixn num_blocks = ceil((double)num_chunks/dMaxThreadsPerBlock);
		if(threadIdx.x == 0) {
			//printf("hit\n");
		} 
		cudaBitAddBNKernel<<<num_blocks, num_threads>>>(value, irred_poly, num_chunks);
	} else {
		if(threadIdx.x == 0) {
			//printf("no hit\n");
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Checks if a is not 0
//
////////////////////////////////////////////////////////////////////////////////
__device__ bool cudaBitCheckBN( sfixn* x, sfixn num_chunks ) {

	sfixn result = false;

	for( sfixn i=0; i<num_chunks; i++ ) {
		if( x[i] > 0 ) {
			result = true;
			break;
		}
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Shift a 1 Bit to the left
//
////////////////////////////////////////////////////////////////////////////////
__device__ void cudaBitShiftLeft1BN( sfixn* a, sfixn num_chunks ) {
	longfixnum tmp = 0;
	sfixn carry = 0;

	longfixnum lmask = pow((double)2, (double)SIZE_CHUNK) - 1;
	longfixnum umask = pow((double)2, (double)SIZE_CHUNK);

	for( sfixn i = num_chunks - 1; i >= 0; --i ) {
		tmp = 0;
		tmp = a[i];
		tmp <<= 1;
		a[i] = (tmp&lmask) | carry;
		carry = (tmp&umask) >> SIZE_CHUNK;
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Calculates the number of chunks for the diggit length
//
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ sfixn getNumberChunks( sfixn length ) {

	if( !length ) {
		return 0;
	} else {
		return (length-1)/SIZE_CHUNK+1;
	}
} 

////////////////////////////////////////////////////////////////////////////////
//
//	Checks if the bit at pos bitnum is set (idx from right to left)
//
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ sfixn isbitset( sfixn val, sfixn bitnum ) {
	return (val & (1 << bitnum)) != 0;
}

__global__ void cudaPrintbincharpadKernel(sfixn* ca, unsigned int n)
{
	for(int j=0; j<n; j++) {
		sfixn c = ca[j];
	    for (int i = sizeof(sfixn)*8-1; i >= 0; --i)
	    {
	        if(c & (1 << i)) 
			printf("%c", '1');
		else
			printf("%c", '0');
	    }
	    printf("%c", ' ');
	}
	printf("\n");
}

__device__ void cudaPrintbincharpad(sfixn* ca, unsigned int n)
{
	for(int j=0; j<n; j++) {
		sfixn c = ca[j];
	    for (int i = sizeof(sfixn)*8-1; i >= 0; --i)
	    {
	        if(c & (1 << i)) 
			printf("%c", '1');
		else
			printf("%c", '0');
	    }
	    printf("%c", ' ');
	}
	printf("\n");
}
