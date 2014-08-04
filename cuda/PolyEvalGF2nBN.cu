#define INCLUDE_FROM_CUDA_FILE
#include "PolyEvalGF2nBN.cuh"
#undef INCLUDE_FROM_CUDA_FILE

////////////////////////////////////////////////////////////////////////////////
/*
	Some constant memory variables
*/
////////////////////////////////////////////////////////////////////////////////
// Device side
__constant__ sfixn dMaxThreadsPerBlock;
__constant__ sfixn dSharedMemPerBlock;

// Host side
static sfixn hMaxThreadsPerBlock;
static sfixn hSharedMemPerBlock;


////////////////////////////////////////////////////////////////////////////////
/*
	Host Functions
*/
////////////////////////////////////////////////////////////////////////////////
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
void evaluateGF2nPolyBN(
	sfixn* coeffs, 
	sfixn* x, 
	sfixn size_field,
	sfixn deg_poly, 
	sfixn* irred_poly,
	sfixn* mask,
	sfixn* result ) {

	// How many coefficients does our polynomial have
	sfixn count_coeffs = deg_poly + 1;

	// Calculate the number of chunks of the data elements
	sfixn num_chunks = getNumberChunks(size_field + 1);

	// Calcualte the amount of leaves needed for used binary tree method
	sfixn width_binary_tree = getCeilToPotOf2n(count_coeffs);

	// How many bytes do we need for one field element
	sfixn bytes_for_chunks = num_chunks * SIZE_CHUNK / SIZE_BYTE;

	// Lead device properties to constant memory
	loadPoroperties();

	// Define all device variables
	sfixn *dx, *dCoeffs, *dIrred_poly, *dMask, *dTmp_Result;
	
	// Allocate the device variables
	cudaMalloc((sfixn**)&dx, bytes_for_chunks);
	cudaMalloc((sfixn**)&dCoeffs, width_binary_tree * bytes_for_chunks);
	cudaMalloc((sfixn**)&dIrred_poly, bytes_for_chunks);
	cudaMalloc((sfixn**)&dMask, bytes_for_chunks);
	cudaMalloc((sfixn**)&dTmp_Result, width_binary_tree * bytes_for_chunks);

	sfixn hCoeffs[width_binary_tree*num_chunks*SIZE_CHUNK/SIZE_BYTE];
	padWithZeros(coeffs, count_coeffs, num_chunks, hCoeffs, width_binary_tree);

	// Copy host data to device
	cudaMemcpy(dx, x, bytes_for_chunks, cudaMemcpyHostToDevice);
	cudaMemcpy(dCoeffs, hCoeffs, width_binary_tree * bytes_for_chunks, cudaMemcpyHostToDevice);
	cudaMemcpy(dIrred_poly, irred_poly, bytes_for_chunks, cudaMemcpyHostToDevice);
	cudaMemcpy(dMask, mask, bytes_for_chunks, cudaMemcpyHostToDevice);

	// Create all exponentiation from x^0 to x^deg_poly and store it in res 
	cudaCreateExpTreeBNKernel<<<min(width_binary_tree/2, hMaxThreadsPerBlock), ceil((double)width_binary_tree/2/hMaxThreadsPerBlock), width_binary_tree/2*num_chunks>>>(dx, num_chunks, deg_poly, dIrred_poly, dMask, width_binary_tree, dTmp_Result);

	// Multiply each coefficient with its related exponentiation of x 
	// (coeff[0]*x^0, ..., coeff[deg_poly]*x^deg_poly) and store it in res
	cudaMontgMulBNKernel<<<min(width_binary_tree, hMaxThreadsPerBlock), ceil((double)width_binary_tree/hMaxThreadsPerBlock)>>>(dCoeffs, dTmp_Result, width_binary_tree, num_chunks, dIrred_poly, dTmp_Result);

	// Add all summands of the polynom up to the result
	cudaBitSumBNKernel<<<min(width_binary_tree/2, hMaxThreadsPerBlock), ceil((double)width_binary_tree/2/hMaxThreadsPerBlock)>>>(dTmp_Result, num_chunks, width_binary_tree);

	// Read the result from device
	sfixn hTmp_Result[width_binary_tree * bytes_for_chunks];
	cudaMemcpy(hTmp_Result, dTmp_Result, width_binary_tree * bytes_for_chunks, cudaMemcpyDeviceToHost);

	// Create result
	for( sfixn i=0; i<num_chunks; ++i ) {
		result[i] = hTmp_Result[i];
	}

	// Free all allocated device memory
	cudaFree(dx);
	cudaFree(dCoeffs);
	cudaFree(dIrred_poly);
	cudaFree(dMask);
	cudaFree(dTmp_Result);
}

////////////////////////////////////////////////////////////////////////////////
//
//	Load some cuda properties to constant memory
//
////////////////////////////////////////////////////////////////////////////////
__host__ void loadPoroperties() {

	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemcpyToSymbol(&dMaxThreadsPerBlock, &deviceProp.maxThreadsPerBlock, sizeof(sfixn));
	cudaMemcpyToSymbol(&dSharedMemPerBlock, &deviceProp.sharedMemPerBlock, sizeof(sfixn));

	hMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	hSharedMemPerBlock = deviceProp.sharedMemPerBlock;
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
			data_new[i] = data_old[i - (size_old - size_old)];
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
__global__ void cudaCreateExpTreeBNKernel(
	sfixn* x, 
	sfixn num_chunks,
	sfixn deg_poly,
	sfixn* irred_poly,
	sfixn* mask, 
	sfixn length_exp_tree,
	sfixn* res) {

    sfixn thid = threadIdx.x * blockIdx.x;

    if( thid < length_exp_tree/2 ) {

	    extern __shared__ sfixn* shared;
;
		sfixn *tmp = &shared[thid * num_chunks];			

		// Fill res with x (Bsp size_field = 3: [x0, x1, x2, x0, x1, x2, ...])
		cudaExpandVecBNKernel<<<min(num_chunks*length_exp_tree, dMaxThreadsPerBlock), ceil((double)num_chunks*length_exp_tree/dMaxThreadsPerBlock)>>>(x, num_chunks, res, num_chunks*length_exp_tree);

	    sfixn offset = 1;

	 	for( sfixn d=length_exp_tree>>1; d>0; d>>=1 ) { //build sum in place up the tree 
	 		__syncthreads();

	 		if( thid < d ) {
	 			sfixn ai = (offset * (2*thid+1) - offset) * num_chunks;
	 			sfixn bi = (offset * (2*thid+2) - offset) * num_chunks;

	 			// Calculate temp_bi = temp_ai * temp_bi
				cudaMontgMulBN(
					&res[ai], 
					&res[bi],
					num_chunks,
					irred_poly,
					&res[ai]);
	 		}

	 		offset <<= 1;
		}

		if( thid == 0 )
			cudaSet0Kernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(&res[0], num_chunks);

		for( sfixn d=1; d<deg_poly; d*=2 ) { //traverse down tree & build scan 

		    offset >>= 1;
		    __syncthreads();

			if( thid < d ) {
				sfixn ai = (offset * (2*thid+1) - offset) * num_chunks;
		 		sfixn bi = (offset * (2*thid+2) - offset) * num_chunks;

		        cudaCopyBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(&res[bi], num_chunks, tmp, num_chunks);
		        cudaCopyBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(&res[ai], num_chunks, &res[bi], num_chunks);
		        cudaBitAddBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(&res[ai], tmp, num_chunks);
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

	sfixn thid = threadIdx.x * blockIdx.x;

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

	sfixn thid = threadIdx.x * blockIdx.x;

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

	sfixn thid = threadIdx.x * blockIdx.x;

	if( thid < num_chunks_b ) {
		if( thid < num_chunks_a )
			b[threadIdx.x] = a[threadIdx.x];
		else
			b[threadIdx.x] = 0;
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
	sfixn* res) {

	sfixn thid = threadIdx.x * blockIdx.x;

	if( thid < num_values ) {

		for( sfixn i=num_chunks-1; i>=0; --i ) {

			sfixn bitnum = 0;

			for( sfixn j=0; j<SIZE_CHUNK; ++j ) {

				if( isbitset(values[thid*num_chunks+i], bitnum) ) 
					cudaBitAddBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(&res[thid*num_chunks], &x_preCalc[thid*num_chunks], num_chunks);

				cudaBitShiftLeft1BN(&x_preCalc[thid*num_chunks], num_chunks);
				cudaReducePolyBN(&x_preCalc[thid*num_chunks], num_chunks, irred_poly, num_chunks);

				++bitnum;
			}
		}
	}
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

	sfixn thid = threadIdx.x * blockIdx.x;

	if( thid < n/2 ) {
		sfixn offset = 1;

		for( sfixn d=n>>1; d>0; d>>=1 ) {
	 		__syncthreads();

	 		if( thid < d ) {
	 			sfixn ai = (offset * (2*thid+1) - offset) * num_chunks;
	 			sfixn bi = (offset * (2*thid+2) - offset) * num_chunks;

	 			// Calculate values_bi = values_ai + values_bi
				cudaBitAddBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(&values[bi], &values[ai], num_chunks);
	 		}

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

	sfixn thid = threadIdx.x * blockIdx.x;

	if( thid < blocks_value_vec )
		value_vec[threadIdx.x] = value[threadIdx.x % blocks_per_value];
}

////////////////////////////////////////////////////////////////////////////////
//
//	Sets x to 0
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaSet0Kernel( sfixn* x, sfixn length ) {

	sfixn thid = threadIdx.x * blockIdx.x;

	if( thid < length )
		x[threadIdx.x] = 0;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in a
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitAddBNKernel(sfixn* a, sfixn* b, sfixn num_chunks) {

	sfixn thid = threadIdx.x * blockIdx.x;

	if( thid < num_chunks )
		a[thid] ^= b[thid];
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in c
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitAndBNKernel(sfixn* a, sfixn* b, sfixn* c, sfixn num_chunks) {

	sfixn thid = threadIdx.x * blockIdx.x;

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
	sfixn* res) {

	for( sfixn i=num_chunks-1; i>=0; --i ) {

		sfixn bitnum = 0;

		for( sfixn j=0; j<SIZE_CHUNK; ++j ) {

			if( isbitset(b[i], bitnum) ) 
				cudaBitAddBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(res, a, num_chunks);

			cudaBitShiftLeft1BN(a, num_chunks);
			cudaReducePolyBN(a, num_chunks, irred_poly, num_chunks);

			++bitnum;
		}

	}
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
	sfixn pos_msb
	) {

	if( isbitset(value[0], pos_msb) ) 
		cudaBitAddBNKernel<<<min(num_chunks, dMaxThreadsPerBlock), ceil((double)num_chunks/dMaxThreadsPerBlock)>>>(value, irred_poly, num_chunks);
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
	longfixnum umask = (pow((double)2, (double)2*SIZE_CHUNK) - 1) - lmask;

	for( sfixn i = num_chunks - 1; i >= 0; --i ) {
		tmp = a[i] << 1;
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