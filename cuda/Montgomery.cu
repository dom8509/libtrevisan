#include "Montgomery.cuh"

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
//		coeffs 			- the coefficients of the polynomial
//		x 				- point where the polynomial is evaluated
//		size_field  	- number of bits of the field elements
//		deg_poly 		- degree of the evaluated polynom
//		irred_poly 		- the irreducible polynomial of the GF(2n)
// 		mask 			- mask of the field
//		result			- the result
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

	// Allocate device data
	sfixn *dx, *dCoeffs, *dIrred_poly, *dMask, *dTmp_Result;
	sfixn bytes_for_chunks = num_chunks*SIZE_CHUNK/SIZE_BYTE;
	cudaMalloc((sfixn**)&dx, bytes_for_chunks);
	cudaMalloc((sfixn**)&dCoeffs, count_coeffs*bytes_for_chunks);
	cudaMalloc((sfixn**)&dIrred_poly, bytes_for_chunks);
	cudaMalloc((sfixn**)&dMask, bytes_for_chunks);
	cudaMalloc((sfixn**)&dTmp_Result, count_coeffs*bytes_for_chunks);

	// Copy host data to device
	cudaMemcpy(dx, x, bytes_for_chunks, cudaMemcpyHostToDevice);
	cudaMemcpy(dCoeffs, coeffs, count_coeffs*bytes_for_chunks, cudaMemcpyHostToDevice);
	cudaMemcpy(dIrred_poly, irred_poly, bytes_for_chunks, cudaMemcpyHostToDevice);
	cudaMemcpy(dMask, mask, bytes_for_chunks, cudaMemcpyHostToDevice);

	// Create all exponentiation from x^0 to x^deg_poly and store it in res 
	sfixn sharedMemory_ExpTree = 3*num_chunks + width_binary_tree*num_chunks;
	cudaCreateExpTreeBNKernel<<<width_binary_tree/2, 1, sharedMemory_ExpTree>>>(dx, size_field, deg_poly, dIrred_poly, dMask, width_binary_tree, dTmp_Result);

	// Multiply each coefficient with its related exponentiation of x 
	// (coeff[0]*x^0, ..., coeff[deg_poly]*x^deg_poly) and store it in res
	cudaMontgMulBNKernel<<<deg_poly+1, 1, 2*num_chunks>>>(dCoeffs, dTmp_Result, size_field, dIrred_poly, size_field+1, dMask, dTmp_Result);

	sfixn hTmp_Result[count_coeffs*bytes_for_chunks];
	cudaMemcpy(hTmp_Result, dTmp_Result, count_coeffs*bytes_for_chunks, cudaMemcpyDeviceToHost);

	// Create leaves for sub sum tree
	sfixn hSummands[width_binary_tree*num_chunks*SIZE_CHUNK/SIZE_BYTE];
	padWithZeros(hTmp_Result, count_coeffs, size_field, hSummands, width_binary_tree);

	// Copy sub sum tree leaves to device
	sfixn* dSummands;
	cudaMalloc((sfixn**)&dSummands, width_binary_tree*num_chunks*SIZE_CHUNK/SIZE_BYTE);
	cudaMemcpy(dSummands, hSummands, width_binary_tree*num_chunks*SIZE_CHUNK/SIZE_BYTE, cudaMemcpyHostToDevice);

	// Add all summands of the polynom up to the result
	cudaBitSumBNKernel<<<width_binary_tree, 1>>>(dSummands, num_chunks, width_binary_tree);

	// Read the result from device
	cudaMemcpy(hSummands, dSummands, width_binary_tree*num_chunks*SIZE_CHUNK/SIZE_BYTE, cudaMemcpyDeviceToHost);

	// Create result
	for( sfixn i=0; i<num_chunks; ++i ) {
		result[i] = hSummands[(width_binary_tree - 1) * num_chunks + i];
	}

	// Free all allocated device memory
	cudaFree(dx);
	cudaFree(dCoeffs);
	cudaFree(dIrred_poly);
	cudaFree(dMask);
	cudaFree(dTmp_Result);
	cudaFree(dSummands);
}

__host__ void padWithZeros( 
	sfixn* data_old, 
	sfixn size_old, 
	sfixn size_field, 
	sfixn* data_new, 
	sfixn size_new ) {

	for( sfixn i=size_old*size_field-1; i>=0; --i ) {
		if( i >= size_old - size_old ) {
			data_new[i] = data_old[i - (size_old - size_old)];
		} else {
			data_new[i] = 0;
		}
	}
}

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
	sfixn size_field,
	sfixn deg_poly,
	sfixn* irred_poly,
	sfixn* mask, 
	sfixn length_exp_tree,
	sfixn* res) {

    sfixn thid = threadIdx.x;

    extern __shared__ sfixn* shared;

	sfixn num_chunks_x = getNumberChunks(size_field);
	sfixn *tmp1 = &shared[0];												// <- TODO: überschreibt nicht jeder kernel den anderen da nicht const???
	sfixn *tmp2 = &shared[num_chunks_x];
	sfixn *xTile = &shared[2*num_chunks_x];									// shared memory for x
	sfixn *leaves = &shared[2*num_chunks_x+num_chunks_x*length_exp_tree];	// shared memory for res

	// copy x to shared memory xTile
	cudaCopyBNKernel<<<num_chunks_x, 1>>>(x, num_chunks_x, xTile, num_chunks_x);

	// Fill leaves with x (Bsp size_field = 3: [x0, x1, x2, x0, x1, x2, ...])
	cudaExpandVecBNKernel<<<num_chunks_x*length_exp_tree, 1>>>(x, size_field, leaves, num_chunks_x*length_exp_tree);

    sfixn offset = 1;

 	for( sfixn d=length_exp_tree>>1; d>0; d>>=1 ) { //build sum in place up the tree 
 		__syncthreads();

 		if( thid < d ) {
 			sfixn ai = (offset * (2*thid+1) - 1) * size_field;
 			sfixn bi = (offset * (2*thid+2) - 1) * size_field;

 			// Calculate temp_bi = temp_ai * temp_bi
			cudaMontgMulBN(
				&leaves[ai], 
				&leaves[bi],
				size_field,
				irred_poly, mask,
				tmp1,
				tmp2,
				&leaves[bi]);
 		}

 		offset <<= 1;
	}

	if( thid == 0 )
		cudaSet0Kernel<<<num_chunks_x, 1>>>(&leaves[deg_poly*size_field-1], num_chunks_x);

	for( sfixn d=1; d<deg_poly; d*=2 ) { //traverse down tree & build scan 

	    offset >>= 1;
	    __syncthreads();

		if( thid < d ) {
			sfixn ai = (offset * (2*thid+1) - 1) * size_field;
	 		sfixn bi = (offset * (2*thid+2) - 1) * size_field;

	        cudaCopyBNKernel<<<num_chunks_x, 1>>>(&leaves[ai], num_chunks_x, tmp1, num_chunks_x);
	        cudaCopyBNKernel<<<num_chunks_x, 1>>>(&leaves[bi], num_chunks_x, &leaves[ai], num_chunks_x);
	        cudaBitAddBNKernel<<<num_chunks_x, 1>>>(&leaves[bi], tmp1, num_chunks_x);
	    }
	}

	__syncthreads(); 

	// write result back to global memeory
	cudaCopyBNKernel<<<num_chunks_x, 1>>>(&leaves[2*thid*size_field], num_chunks_x, &res[2*thid*size_field], num_chunks_x);
	cudaCopyBNKernel<<<num_chunks_x, 1>>>(&leaves[(2*thid+2)*size_field], num_chunks_x, &res[(2*thid+1)*size_field], num_chunks_x);
}

__global__ void cudaBitShiftRightBNKernel(sfixn* a, sfixn length_a, sfixn n, sfixn* c) {

	sfixn thid = threadIdx.x;

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
				sfixn mask_b = (2^SIZE_CHUNK-1) - (2^(SIZE_CHUNK-(max_src_range%SIZE_CHUNK))-1);
				sfixn src_idx_b = max_src_range/SIZE_CHUNK;
				c[thid] = ((a[src_idx_b]&mask_b) >> (SIZE_CHUNK-(max_src_range%SIZE_CHUNK)));
			} else if(digits_block_b_shift == 0) {
				c[thid] = a[min_src_range/SIZE_CHUNK];
			} else {
				sfixn mask_a = 2^(SIZE_CHUNK-(min_src_range%SIZE_CHUNK))-1;
				sfixn mask_b = (2^SIZE_CHUNK-1) - (2^(SIZE_CHUNK-(max_src_range%SIZE_CHUNK))-1);

				sfixn src_idx_a = min_src_range/SIZE_CHUNK;
				sfixn src_idx_b = max_src_range/SIZE_CHUNK;

				c[thid] = ((a[src_idx_a]&mask_a) << (min_src_range%SIZE_CHUNK)) | 
					((a[src_idx_b]&mask_b) >> (SIZE_CHUNK-(max_src_range%SIZE_CHUNK)));

			}
		}
	}
}

__global__ void cudaBitShiftLeftBNKernel(sfixn* a, sfixn num_chunks, sfixn n, sfixn* c) {

	sfixn thid = threadIdx.x;

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
				sfixn mask_b = 2^(SIZE_CHUNK-(min_src_range % SIZE_CHUNK)) - 1;
				sfixn src_idx_b = min_src_range/SIZE_CHUNK;
				c[thid] = ((a[src_idx_b]&mask_b) << (min_src_range%SIZE_CHUNK));
			} else if( shift_whole_block ) {
				c[thid] = a[min_src_range/SIZE_CHUNK];
			} else {
				sfixn mask_a = 2^(SIZE_CHUNK-(min_src_range%SIZE_CHUNK))-1;
				sfixn mask_b = (2^SIZE_CHUNK-1) - (2^(SIZE_CHUNK-((max_src_range+1)%SIZE_CHUNK))-1);

				sfixn src_idx_a = min_src_range/SIZE_CHUNK;
				sfixn src_idx_b = max_src_range/SIZE_CHUNK;

				c[thid] = ((a[src_idx_a]&mask_a) << (min_src_range%SIZE_CHUNK)) | 
					((a[src_idx_b]&mask_b) >> (SIZE_CHUNK-((max_src_range+1)%SIZE_CHUNK)));

			}
		}
	}
}

__global__ void cudaCopyBNKernel( sfixn* a, sfixn num_chunks_a, sfixn* b, sfixn num_chunks_b ) {

	if(threadIdx.x < num_chunks_b) {
		if(threadIdx.x < num_chunks_a)
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
	sfixn field_size, 
	sfixn* irred_poly, sfixn length_irred_poly, sfixn* irred_poly_mask, 
	sfixn* res) {

	sfixn thid = threadIdx.x;

	extern __shared__ sfixn *shared;

	sfixn num_chunks = getNumberChunks(field_size+1);

	sfixn *tmp_a = &shared[0];
	sfixn *tmp_b = &shared[num_chunks];
	cudaCopyBNKernel<<<num_chunks, 1>>>(&x_preCalc[thid*field_size], num_chunks, tmp_a, num_chunks);

	for( sfixn i=num_chunks-1; i>=0; --i ) {

		sfixn mask = 1;

		for( sfixn j=0; j<SIZE_CHUNK; ++j ) {

			if( values[thid*field_size+i]&mask > 0 ) 
				cudaBitAddBNKernel<<<num_chunks, 1>>>(&res[thid*field_size], tmp_a, num_chunks);

			cudaBitShiftLeftBNKernel<<<num_chunks, 1>>>(tmp_a, num_chunks, 1, tmp_b);
			cudaReducePolyBN(tmp_b, num_chunks, irred_poly, irred_poly_mask, tmp_a);
			cudaCopyBNKernel<<<num_chunks, 1>>>(tmp_b, num_chunks, tmp_a, num_chunks);

			mask <<= 1;
		}
	}
}

__global__ void cudaBitSumBNKernel(
	sfixn* values, 
	sfixn field_size,
	sfixn n ) {

	sfixn thid = threadIdx.x;

	sfixn num_chunks = getNumberChunks(field_size+1);

	sfixn offset = 1;

	for( sfixn d=n>>1; d>0; d>>=1 ) {
 		__syncthreads();

 		if( thid < d ) {
 			sfixn ai = (offset * (2*thid+1) - 1) * field_size;
 			sfixn bi = (offset * (2*thid+2) - 1) * field_size;

 			// Calculate values_bi = values_ai + values_bi
			cudaBitAddBNKernel<<<num_chunks, 1>>>(&values[bi], &values[ai], num_chunks);
 		}

 		offset <<= 1;
	}
}

__global__ void cudaExpandVecBNKernel(
	sfixn* value,
	sfixn blocks_per_value,
	sfixn* value_vec,
	sfixn blocks_value_vec
	) {

	if( threadIdx.x < blocks_value_vec )
		value_vec[threadIdx.x] = value[threadIdx.x % blocks_per_value];
}

__global__ void cudaSet0Kernel( sfixn* x, sfixn length ) {
	if( threadIdx.x < length )
		x[threadIdx.x] = 0;
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in a
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitAddBNKernel(sfixn* a, sfixn* b, sfixn num_chunks) {

	sfixn thid = threadIdx.x;

	if( thid < num_chunks )
		a[thid] ^= b[thid];
}

////////////////////////////////////////////////////////////////////////////////
//
//	Adds a and b and stores the result in c
//
////////////////////////////////////////////////////////////////////////////////
__global__ void cudaBitAndBNKernel(sfixn* a, sfixn* b, sfixn* c, sfixn num_chunks) {

	sfixn thid = threadIdx.x;

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
	sfixn size_field,
	sfixn* irred_poly, 
	sfixn* irred_poly_mask, 
	sfixn* tmp_a, sfixn* tmp_b,
	sfixn* res) {

	sfixn mask = 1;
	sfixn num_chunks = getNumberChunks(size_field+1);

	cudaCopyBNKernel<<<num_chunks, 1>>>(a, num_chunks, tmp_a, num_chunks);

	for( sfixn i=num_chunks-1; i>=0; --i ) {

		for( sfixn j=0; j<SIZE_CHUNK; ++j ) {

			if( b[i]&mask > 0 )
				cudaBitAddBNKernel<<<num_chunks, 1>>>(res, tmp_a, num_chunks);

			cudaBitShiftLeftBNKernel<<<num_chunks, 1>>>(tmp_a, num_chunks, 1, tmp_b);
			cudaReducePolyBN(tmp_b, num_chunks, irred_poly, irred_poly_mask, tmp_a);
			cudaCopyBNKernel<<<num_chunks, 1>>>(tmp_b, num_chunks, tmp_a, num_chunks);

			mask <<= 1;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
//	Reduce the extended field polynomial
//
////////////////////////////////////////////////////////////////////////////////
__device__ void cudaReducePolyBN(
	sfixn* coeffs,
	sfixn num_chunks_coeffs,
	sfixn* irred_poly,
	sfixn* irred_poly_mask,
	sfixn* tmp
	) {

	cudaBitAndBNKernel<<<num_chunks_coeffs, 1>>>(coeffs, irred_poly_mask, tmp, num_chunks_coeffs);
	if( cudaBitCheckBN(tmp, num_chunks_coeffs) )
		cudaBitAddBNKernel<<<num_chunks_coeffs, 1>>>(coeffs, irred_poly, num_chunks_coeffs);
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